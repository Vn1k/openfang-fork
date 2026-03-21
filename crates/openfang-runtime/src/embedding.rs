//! Embedding driver for vector-based semantic memory.
//!
//! Provides an `EmbeddingDriver` trait and an OpenAI-compatible implementation
//! that works with any provider offering a `/v1/embeddings` endpoint (OpenAI,
//! Groq, Together, Fireworks, Ollama, etc.).
//!
//! ## Circular dependency fix
//!
//! The `EmbeddingDriver` trait and `EmbeddingError` are now defined in
//! `openfang-types::driver::embedding` and re-exported here.  All existing
//! callers that import from `openfang_runtime::embedding` continue to compile
//! without any changes.
//!
//! What stays in this file:
//! - Re-exports of the canonical trait/error from `openfang-types`
//! - `EmbeddingConfig` — driver construction detail
//! - `OpenAIEmbeddingDriver` — the concrete HTTP implementation
//! - `create_embedding_driver()` — factory function
//! - Utility functions: `cosine_similarity`, `embedding_to_bytes`, `embedding_from_bytes`

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports from openfang-types (single source of truth)
// ─────────────────────────────────────────────────────────────────────────────

/// Re-export the canonical `EmbeddingDriver` trait from `openfang-types`.
///
/// All callers that previously wrote
/// `use openfang_runtime::embedding::EmbeddingDriver` continue to compile.
pub use openfang_types::driver::embedding::{EmbeddingDriver, EmbeddingError};

// ─────────────────────────────────────────────────────────────────────────────
// Local imports
// ─────────────────────────────────────────────────────────────────────────────

use async_trait::async_trait;
use openfang_types::model_catalog::{
    FIREWORKS_BASE_URL, GROQ_BASE_URL, LMSTUDIO_BASE_URL, MISTRAL_BASE_URL, OLLAMA_BASE_URL,
    OPENAI_BASE_URL, TOGETHER_BASE_URL, VLLM_BASE_URL,
};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};
use zeroize::Zeroizing;

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for constructing an embedding driver.
///
/// Stays in `openfang-runtime` — it is an implementation detail only needed
/// during driver construction.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Provider name (e.g. `"openai"`, `"ollama"`, `"groq"`).
    pub provider: String,
    /// Model name (e.g. `"text-embedding-3-small"`, `"all-MiniLM-L6-v2"`).
    pub model: String,
    /// API key resolved from the environment variable.
    pub api_key: String,
    /// Base URL for the provider's `/v1/embeddings` endpoint.
    pub base_url: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenAIEmbeddingDriver
// ─────────────────────────────────────────────────────────────────────────────

/// OpenAI-compatible embedding driver.
///
/// Works with any provider that implements `/v1/embeddings`:
/// OpenAI, Groq, Together, Fireworks, Ollama, vLLM, LM Studio, etc.
///
/// Stays in `openfang-runtime` because it depends on `reqwest` and `zeroize`
/// which are not available in `openfang-types`.
pub struct OpenAIEmbeddingDriver {
    api_key: Zeroizing<String>,
    base_url: String,
    model: String,
    client: reqwest::Client,
    dims: usize,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

impl OpenAIEmbeddingDriver {
    /// Create a new OpenAI-compatible embedding driver from config.
    pub fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        let dims = infer_dimensions(&config.model);
        Ok(Self {
            api_key: Zeroizing::new(config.api_key),
            base_url: config.base_url,
            model: config.model,
            client: reqwest::Client::new(),
            dims,
        })
    }
}

/// Infer embedding dimensions from a model name.
fn infer_dimensions(model: &str) -> usize {
    match model {
        // OpenAI
        "text-embedding-3-small" => 1536,
        "text-embedding-3-large" => 3072,
        "text-embedding-ada-002" => 1536,
        // Sentence Transformers / local models
        "all-MiniLM-L6-v2" => 384,
        "all-MiniLM-L12-v2" => 384,
        "all-mpnet-base-v2" => 768,
        "nomic-embed-text" => 768,
        "mxbai-embed-large" => 1024,
        // Default fallback (most common OpenAI dimension)
        _ => 1536,
    }
}

// NOTE: `EmbeddingDriver` here refers to the re-exported trait from
// `openfang-types::driver::embedding`.  The impl is unchanged.
#[async_trait]
impl EmbeddingDriver for OpenAIEmbeddingDriver {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let url = format!("{}/embeddings", self.base_url);
        let body = EmbedRequest {
            model: &self.model,
            input: texts,
        };

        let mut req = self.client.post(&url).json(&body);
        if !self.api_key.as_str().is_empty() {
            req = req.header(
                "Authorization",
                format!("Bearer {}", self.api_key.as_str()),
            );
        }

        let resp = req
            .send()
            .await
            .map_err(|e| EmbeddingError::Http(e.to_string()))?;

        let status = resp.status().as_u16();
        if status != 200 {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(EmbeddingError::Api {
                status,
                message: body_text,
            });
        }

        let data: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| EmbeddingError::Parse(e.to_string()))?;

        let embeddings: Vec<Vec<f32>> = data.data.into_iter().map(|d| d.embedding).collect();

        debug!(
            "Embedded {} texts (dims={})",
            embeddings.len(),
            embeddings.first().map(|e| e.len()).unwrap_or(0)
        );

        Ok(embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────

/// Create an embedding driver from kernel config parameters.
///
/// Resolves the API key from the named environment variable, determines the
/// correct base URL for the provider, and returns a boxed `EmbeddingDriver`.
pub fn create_embedding_driver(
    provider: &str,
    model: &str,
    api_key_env: &str,
    custom_base_url: Option<&str>,
) -> Result<Box<dyn EmbeddingDriver + Send + Sync>, EmbeddingError> {
    let api_key = if api_key_env.is_empty() {
        String::new()
    } else {
        std::env::var(api_key_env).unwrap_or_default()
    };

    let base_url = custom_base_url
        .filter(|u| !u.is_empty())
        .map(|u| {
            let trimmed = u.trim_end_matches('/');
            // All OpenAI-compatible embedding providers expect /v1 in the path.
            // Append it if the user supplied a bare host URL.
            let needs_v1 = matches!(
                provider,
                "openai"
                    | "groq"
                    | "together"
                    | "fireworks"
                    | "mistral"
                    | "ollama"
                    | "vllm"
                    | "lmstudio"
            );
            if needs_v1 && !trimmed.ends_with("/v1") {
                format!("{trimmed}/v1")
            } else {
                trimmed.to_string()
            }
        })
        .unwrap_or_else(|| match provider {
            "openai" => OPENAI_BASE_URL.to_string(),
            "groq" => GROQ_BASE_URL.to_string(),
            "together" => TOGETHER_BASE_URL.to_string(),
            "fireworks" => FIREWORKS_BASE_URL.to_string(),
            "mistral" => MISTRAL_BASE_URL.to_string(),
            "ollama" => OLLAMA_BASE_URL.to_string(),
            "vllm" => VLLM_BASE_URL.to_string(),
            "lmstudio" => LMSTUDIO_BASE_URL.to_string(),
            other => {
                warn!("Unknown embedding provider '{other}', using OpenAI-compatible format");
                format!("https://{other}/v1")
            }
        });

    // SECURITY: warn when text will be sent to an external API
    let is_local = base_url.contains("localhost")
        || base_url.contains("127.0.0.1")
        || base_url.contains("[::1]");
    if !is_local {
        warn!(
            provider = %provider,
            base_url = %base_url,
            "Embedding driver will send text content to an external API"
        );
    }

    let config = EmbeddingConfig {
        provider: provider.to_string(),
        model: model.to_string(),
        api_key,
        base_url,
    };

    Ok(Box::new(OpenAIEmbeddingDriver::new(config)?))
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions (unchanged from original)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute cosine similarity between two embedding vectors.
///
/// Returns a value in `[-1.0, 1.0]` where `1.0` means identical direction.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

/// Serialize an embedding vector to bytes for SQLite `BLOB` storage.
pub fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Deserialize an embedding vector from SQLite `BLOB` bytes.
pub fn embedding_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (unchanged from original)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_real_vectors() {
        let a = vec![0.1, 0.2, 0.3, 0.4];
        let b = vec![0.1, 0.2, 0.3, 0.4];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);
        let c = vec![0.4, 0.3, 0.2, 0.1];
        let sim2 = cosine_similarity(&a, &c);
        assert!(sim2 > 0.0 && sim2 < 1.0);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn test_embedding_roundtrip() {
        let embedding = vec![0.1, -0.5, 1.23456, 0.0, -1e10, 1e10];
        let bytes = embedding_to_bytes(&embedding);
        let recovered = embedding_from_bytes(&bytes);
        assert_eq!(embedding.len(), recovered.len());
        for (a, b) in embedding.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_embedding_bytes_empty() {
        assert!(embedding_to_bytes(&[]).is_empty());
        assert!(embedding_from_bytes(&[]).is_empty());
    }

    #[test]
    fn test_infer_dimensions() {
        assert_eq!(infer_dimensions("text-embedding-3-small"), 1536);
        assert_eq!(infer_dimensions("all-MiniLM-L6-v2"), 384);
        assert_eq!(infer_dimensions("nomic-embed-text"), 768);
        assert_eq!(infer_dimensions("unknown-model"), 1536);
    }

    #[test]
    fn test_create_embedding_driver_ollama() {
        let driver = create_embedding_driver("ollama", "all-MiniLM-L6-v2", "", None);
        assert!(driver.is_ok());
        assert_eq!(driver.unwrap().dimensions(), 384);
    }

    #[test]
    fn test_create_embedding_driver_custom_url_with_v1() {
        let driver = create_embedding_driver(
            "ollama",
            "nomic-embed-text",
            "",
            Some("http://192.168.0.1:11434/v1"),
        );
        assert!(driver.is_ok());
    }

    #[test]
    fn test_create_embedding_driver_custom_url_without_v1() {
        let driver = create_embedding_driver(
            "ollama",
            "nomic-embed-text",
            "",
            Some("http://192.168.0.1:11434"),
        );
        assert!(driver.is_ok());
    }

    #[test]
    fn test_create_embedding_driver_custom_url_trailing_slash() {
        let driver = create_embedding_driver(
            "ollama",
            "nomic-embed-text",
            "",
            Some("http://192.168.0.1:11434/"),
        );
        assert!(driver.is_ok());
    }
}