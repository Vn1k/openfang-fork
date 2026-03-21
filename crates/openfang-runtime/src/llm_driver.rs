//! LLM driver trait and types.
//!
//! Abstracts over multiple LLM providers (Anthropic, OpenAI, Ollama, etc.).
//!
//! ## Design
//!
//! `LlmDriver` (including `stream()`) is defined in `openfang-types::driver`
//! so that `openfang-memory` can import it without creating a circular
//! dependency.  Everything is re-exported here so all existing callers that
//! import from `openfang_runtime::llm_driver` continue to compile unchanged.
//!
//! ## Why stream() is in LlmDriver (not a separate trait)
//!
//! `Arc<dyn LlmDriver>` is the primary driver type throughout the codebase.
//! Methods callable on a trait object must be in the trait's vtable.
//! Putting `stream()` in a separate `LlmDriverExt` trait means it cannot be
//! dispatched dynamically — the compiler can't reach the specific driver's
//! implementation.  With `stream()` in `LlmDriver` (with a default fallback
//! impl), every concrete driver can override it and the call dispatches
//! correctly through the vtable.

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports from openfang-types — single source of truth
//
// All callers that import from `openfang_runtime::llm_driver` continue to
// compile without any changes.
// ─────────────────────────────────────────────────────────────────────────────

pub use openfang_types::driver::CompletionRequest;
pub use openfang_types::driver::CompletionResponse;
pub use openfang_types::driver::LlmDriver;
pub use openfang_types::driver::LlmError;
pub use openfang_types::driver::StreamEvent;

// ─────────────────────────────────────────────────────────────────────────────
// DriverConfig
//
// Implementation detail for driver construction — stays in openfang-runtime.
// ─────────────────────────────────────────────────────────────────────────────

use serde::{Deserialize, Serialize};

/// Configuration for constructing an LLM driver instance.
#[derive(Clone, Serialize, Deserialize)]
pub struct DriverConfig {
    /// Provider name (e.g. `"anthropic"`, `"openai"`, `"ollama"`).
    pub provider: String,
    /// API key (resolved from environment variable before construction).
    pub api_key: Option<String>,
    /// Base URL override (uses the catalog default when `None`).
    pub base_url: Option<String>,
    /// Skip interactive permission prompts (Claude Code provider only).
    #[serde(default = "default_skip_permissions")]
    pub skip_permissions: bool,
}

fn default_skip_permissions() -> bool {
    true
}

/// SECURITY: Custom `Debug` redacts the API key.
impl std::fmt::Debug for DriverConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverConfig")
            .field("provider", &self.provider)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("base_url", &self.base_url)
            .field("skip_permissions", &self.skip_permissions)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use openfang_types::message::{ContentBlock, StopReason, TokenUsage};

    #[test]
    fn test_completion_response_text() {
        let response = CompletionResponse {
            content: vec![
                ContentBlock::Text { text: "Hello ".to_string(), provider_metadata: None },
                ContentBlock::Text { text: "world!".to_string(), provider_metadata: None },
            ],
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage::default(),
        };
        assert_eq!(response.text(), "Hello world!");
    }

    #[test]
    fn test_stream_event_clone() {
        let event = StreamEvent::TextDelta { text: "hello".to_string() };
        let cloned = event.clone();
        assert!(matches!(cloned, StreamEvent::TextDelta { text } if text == "hello"));
    }

    #[tokio::test]
    async fn test_default_stream_sends_events() {
        use tokio::sync::mpsc;

        struct FakeDriver;

        #[async_trait]
        impl LlmDriver for FakeDriver {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, LlmError> {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: "Hello!".to_string(),
                        provider_metadata: None,
                    }],
                    stop_reason: StopReason::EndTurn,
                    tool_calls: vec![],
                    usage: TokenUsage { input_tokens: 5, output_tokens: 3 },
                })
            }
        }

        let driver = FakeDriver;
        let (tx, mut rx) = mpsc::channel(16);
        let request = CompletionRequest {
            model: "test".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.0,
            system: None,
            thinking: None,
        };

        // FakeDriver uses the default stream() from LlmDriver
        let response = driver.stream(request, tx).await.unwrap();
        assert_eq!(response.text(), "Hello!");

        let ev1 = rx.recv().await.unwrap();
        assert!(matches!(ev1, StreamEvent::TextDelta { text } if text == "Hello!"));

        let ev2 = rx.recv().await.unwrap();
        assert!(matches!(ev2, StreamEvent::ContentComplete { stop_reason: StopReason::EndTurn, .. }));
    }
}