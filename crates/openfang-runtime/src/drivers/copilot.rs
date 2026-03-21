//! GitHub Copilot authentication — exchanges a GitHub PAT for a Copilot API token.
//!
//! The Copilot API uses the OpenAI chat completions format, so this module
//! handles token exchange and caching, then delegates to the OpenAI-compatible driver.

use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::{debug, warn};
use zeroize::Zeroizing;

const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const TOKEN_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(10);
const REFRESH_BUFFER_SECS: u64 = 300;
pub const GITHUB_COPILOT_BASE_URL: &str = "https://api.githubcopilot.com";

/// Cached Copilot API token with expiry and derived base URL.
#[derive(Clone)]
pub struct CachedToken {
    pub token: Zeroizing<String>,
    pub expires_at: Instant,
    pub base_url: String,
}

impl CachedToken {
    pub fn is_valid(&self) -> bool {
        self.expires_at > Instant::now() + Duration::from_secs(REFRESH_BUFFER_SECS)
    }
}

pub struct CopilotTokenCache {
    cached: Mutex<Option<CachedToken>>,
}

impl CopilotTokenCache {
    pub fn new() -> Self { Self { cached: Mutex::new(None) } }

    pub fn get(&self) -> Option<CachedToken> {
        let lock = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        lock.as_ref().filter(|t| t.is_valid()).cloned()
    }

    pub fn set(&self, token: CachedToken) {
        let mut lock = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        *lock = Some(token);
    }
}

impl Default for CopilotTokenCache {
    fn default() -> Self { Self::new() }
}

pub async fn exchange_copilot_token(github_token: &str) -> Result<CachedToken, String> {
    let client = reqwest::Client::builder()
        .timeout(TOKEN_EXCHANGE_TIMEOUT)
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    debug!("Exchanging GitHub token for Copilot API token");

    let resp = client
        .get(COPILOT_TOKEN_URL)
        .header("Authorization", format!("token {github_token}"))
        .header("Accept", "application/json")
        .header("User-Agent", "OpenFang/1.0")
        .send()
        .await
        .map_err(|e| format!("Copilot token exchange failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Copilot token exchange returned {status}: {body}"));
    }

    let body: serde_json::Value = resp.json().await
        .map_err(|e| format!("Failed to parse Copilot token response: {e}"))?;

    let raw_token = body.get("token").and_then(|v| v.as_str())
        .ok_or("Missing 'token' field in Copilot response")?;

    let expires_at_unix = body.get("expires_at").and_then(|v| v.as_i64()).unwrap_or(0);
    let now_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
    let ttl_secs = (expires_at_unix - now_unix).max(60) as u64;

    let (_, proxy_ep) = parse_copilot_token(raw_token);
    let base_url = proxy_ep.unwrap_or_else(|| GITHUB_COPILOT_BASE_URL.to_string());

    if !base_url.starts_with("https://") {
        warn!(url = %base_url, "Copilot proxy-ep is not HTTPS, using default");
        return Ok(CachedToken {
            token: Zeroizing::new(raw_token.to_string()),
            expires_at: Instant::now() + Duration::from_secs(ttl_secs),
            base_url: GITHUB_COPILOT_BASE_URL.to_string(),
        });
    }

    Ok(CachedToken {
        token: Zeroizing::new(raw_token.to_string()),
        expires_at: Instant::now() + Duration::from_secs(ttl_secs),
        base_url,
    })
}

pub fn parse_copilot_token(raw: &str) -> (String, Option<String>) {
    let mut proxy_ep = None;
    for segment in raw.split(';') {
        if let Some(url) = segment.trim().strip_prefix("proxy-ep=") {
            proxy_ep = Some(url.to_string());
        }
    }
    (raw.to_string(), proxy_ep)
}

pub fn copilot_auth_available() -> bool {
    std::env::var("GITHUB_TOKEN").is_ok()
}

pub struct CopilotDriver {
    github_token: Zeroizing<String>,
    token_cache: CopilotTokenCache,
}

impl CopilotDriver {
    pub fn new(github_token: String, _base_url: String) -> Self {
        Self {
            github_token: Zeroizing::new(github_token),
            token_cache: CopilotTokenCache::new(),
        }
    }

    async fn ensure_token(&self) -> Result<CachedToken, crate::llm_driver::LlmError> {
        if let Some(cached) = self.token_cache.get() {
            return Ok(cached);
        }
        debug!("Copilot token expired or missing, exchanging...");
        let token = exchange_copilot_token(&self.github_token)
            .await
            .map_err(|e| crate::llm_driver::LlmError::Api {
                status: 401,
                message: format!("Copilot token exchange failed: {e}"),
            })?;
        self.token_cache.set(token.clone());
        Ok(token)
    }

    fn make_inner_driver(&self, token: &CachedToken) -> super::openai::OpenAIDriver {
        let base_url = if token.base_url.is_empty() {
            GITHUB_COPILOT_BASE_URL.to_string()
        } else {
            token.base_url.clone()
        };
        super::openai::OpenAIDriver::new(token.token.to_string(), base_url).with_extra_headers(
            vec![
                ("Editor-Version".to_string(), "vscode/1.96.0".to_string()),
                ("Editor-Plugin-Version".to_string(), "copilot/1.250.0".to_string()),
                ("Copilot-Integration-Id".to_string(), "vscode-chat".to_string()),
            ],
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LlmDriver impl — complete() only
// ─────────────────────────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl crate::llm_driver::LlmDriver for CopilotDriver {
    async fn complete(
        &self,
        request: crate::llm_driver::CompletionRequest,
    ) -> Result<crate::llm_driver::CompletionResponse, crate::llm_driver::LlmError> {
        let token = self.ensure_token().await?;
        let driver = self.make_inner_driver(&token);
        driver.complete(request).await
    }

    async fn stream(
        &self,
        request: crate::llm_driver::CompletionRequest,
        tx: tokio::sync::mpsc::Sender<crate::llm_driver::StreamEvent>,
    ) -> Result<crate::llm_driver::CompletionResponse, crate::llm_driver::LlmError> {
        let token = self.ensure_token().await?;
        let driver = self.make_inner_driver(&token);
        driver.stream(request, tx).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_copilot_token_with_proxy() {
        let raw = "tid=abc123;exp=1700000000;sku=copilot_for_individual;proxy-ep=https://copilot-proxy.example.com";
        let (token, proxy) = parse_copilot_token(raw);
        assert_eq!(token, raw);
        assert_eq!(proxy, Some("https://copilot-proxy.example.com".to_string()));
    }

    #[test]
    fn test_parse_copilot_token_without_proxy() {
        let raw = "tid=abc123;exp=1700000000;sku=copilot_for_individual";
        let (_, proxy) = parse_copilot_token(raw);
        assert!(proxy.is_none());
    }

    #[test]
    fn test_token_cache_empty() {
        assert!(CopilotTokenCache::new().get().is_none());
    }

    #[test]
    fn test_token_cache_set_get() {
        let cache = CopilotTokenCache::new();
        cache.set(CachedToken {
            token: Zeroizing::new("test-token".to_string()),
            expires_at: Instant::now() + Duration::from_secs(3600),
            base_url: GITHUB_COPILOT_BASE_URL.to_string(),
        });
        let cached = cache.get();
        assert!(cached.is_some());
        assert_eq!(*cached.unwrap().token, "test-token");
    }

    #[test]
    fn test_token_validity() {
        let valid = CachedToken {
            token: Zeroizing::new("t".to_string()),
            expires_at: Instant::now() + Duration::from_secs(3600),
            base_url: GITHUB_COPILOT_BASE_URL.to_string(),
        };
        assert!(valid.is_valid());

        let almost_expired = CachedToken {
            token: Zeroizing::new("t".to_string()),
            expires_at: Instant::now() + Duration::from_secs(60),
            base_url: GITHUB_COPILOT_BASE_URL.to_string(),
        };
        assert!(!almost_expired.is_valid());
    }
}