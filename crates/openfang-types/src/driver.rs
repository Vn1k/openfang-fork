//! Driver interface traits and shared request/response types.
//!
//! This module lives in `openfang-types` (leaf crate, no upstream deps) so
//! both `openfang-memory` and `openfang-runtime` can import the same types
//! without creating a circular dependency.
//!
//! ## Dependency graph (before)
//! ```text
//! openfang-runtime  (defines LlmDriver, CompletionRequest, …)
//!   └─ openfang-memory  ← needs LlmDriver → must import openfang-runtime
//!        └─ openfang-runtime   ← CIRCULAR ✗
//! ```
//!
//! ## Dependency graph (after)
//! ```text
//! openfang-runtime  (re-exports from openfang-types, keeps StreamEvent + stream())
//!   └─ openfang-memory  (imports from openfang-types) ✓
//!        └─ openfang-types  (owns CompletionRequest, LlmDriver, EmbeddingDriver) ✓
//! ```
//!
//! ## What lives here vs. in `openfang-runtime`
//!
//! | Item | Location | Reason |
//! |---|---|---|
//! | `CompletionRequest` | `openfang-types` | pure data, deps already here |
//! | `CompletionResponse` | `openfang-types` | uses `ContentBlock` etc. already here |
//! | `LlmError` | `openfang-types` | plain error enum |
//! | `LlmDriver` (just `complete()`) | `openfang-types` | `openfang-memory` only calls this |
//! | `StreamEvent` | `openfang-runtime` | needs `tokio` + rich UX types |
//! | `stream()` default impl | `openfang-types` (in `LlmDriver`) | uses tokio mpsc |
//! | `DriverConfig` | `openfang-runtime` | implementation detail |
//! | `EmbeddingError` | `openfang-types` | plain error enum |
//! | `EmbeddingDriver` (trait) | `openfang-types` | only uses `&str` / `Vec<f32>` |
//! | `OpenAIEmbeddingDriver` | `openfang-runtime` | needs `reqwest`, `zeroize` |

use crate::config::ThinkingConfig;
use crate::message::{ContentBlock, Message, StopReason, TokenUsage};
use crate::tool::{ToolCall, ToolDefinition};
use async_trait::async_trait;
use thiserror::Error;


// ─────────────────────────────────────────────────────────────────────────────
// StreamEvent
//
// Defined here (not in openfang-runtime) so that stream() can live in
// LlmDriver and be dispatchable via `dyn LlmDriver` trait objects.
// ─────────────────────────────────────────────────────────────────────────────

/// Events emitted during a streaming LLM completion.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Incremental text content from the model.
    TextDelta { text: String },
    /// A tool-use block has started.
    ToolUseStart { id: String, name: String },
    /// Incremental JSON input for an in-progress tool-use block.
    ToolInputDelta { text: String },
    /// A tool-use block is fully received with parsed input.
    ToolUseEnd {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Incremental thinking/reasoning text.
    ThinkingDelta { text: String },
    /// The entire response is complete.
    ContentComplete {
        stop_reason: crate::message::StopReason,
        usage: crate::message::TokenUsage,
    },
    /// Agent lifecycle phase change (for UX indicators).
    PhaseChange {
        phase: String,
        detail: Option<String>,
    },
    /// Tool execution result (emitted by the agent loop, not the driver).
    ToolExecutionResult {
        name: String,
        result_preview: String,
        is_error: bool,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// CompletionRequest
// ─────────────────────────────────────────────────────────────────────────────

/// A request to an LLM completion endpoint.
///
/// All field types are already defined in `openfang-types`, so this struct
/// compiles with zero new dependencies.
///
/// **Direct equivalent** of `CompletionRequest` in `openfang-runtime/src/llm_driver.rs`.
/// That file re-exports this type so all existing callers compile unchanged.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Model identifier, e.g. `"claude-sonnet-4-20250514"`.
    /// Provider prefix is stripped by the driver resolver before reaching the API.
    pub model: String,

    /// Conversation messages. System messages should be passed via `system`,
    /// not included here (some providers require them separately).
    pub messages: Vec<Message>,

    /// Tool definitions available for this request.
    pub tools: Vec<ToolDefinition>,

    /// Maximum tokens to generate.
    pub max_tokens: u32,

    /// Sampling temperature (0.0 = deterministic, 1.0 = creative).
    pub temperature: f32,

    /// Optional system prompt (extracted from messages for APIs that need it
    /// passed separately, e.g. Anthropic).
    pub system: Option<String>,

    /// Extended thinking configuration (Anthropic Claude only).
    /// `None` disables thinking.
    pub thinking: Option<ThinkingConfig>,
}

// ─────────────────────────────────────────────────────────────────────────────
// CompletionResponse
// ─────────────────────────────────────────────────────────────────────────────

/// A response from an LLM completion.
///
/// Uses the rich `ContentBlock` type (already in `openfang-types`) so the
/// struct is identical to what `openfang-runtime` previously defined locally.
///
/// **Direct equivalent** of `CompletionResponse` in `openfang-runtime/src/llm_driver.rs`.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// All content blocks returned by the model (Text, Thinking, ToolUse, …).
    pub content: Vec<ContentBlock>,

    /// Why the model stopped generating.
    pub stop_reason: StopReason,

    /// Tool calls extracted from the response.
    pub tool_calls: Vec<ToolCall>,

    /// Token usage statistics (for metering / cost tracking).
    pub usage: TokenUsage,
}

impl CompletionResponse {
    /// Concatenate all `Text` content blocks into a single string.
    ///
    /// Thinking blocks are excluded — callers that need them should iterate
    /// `self.content` directly.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Returns `true` if the response contains any meaningful content,
    /// including Thinking blocks.
    ///
    /// Used to distinguish genuine empty responses from thinking-only responses.
    pub fn has_any_content(&self) -> bool {
        self.content.iter().any(|block| match block {
            ContentBlock::Text { text, .. } => !text.is_empty(),
            ContentBlock::Thinking { thinking, .. } => !thinking.is_empty(),
            ContentBlock::ToolUse { .. } | ContentBlock::Image { .. } => true,
            _ => false,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LlmError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur when calling an LLM driver.
///
/// **Direct equivalent** of `LlmError` in `openfang-runtime/src/llm_driver.rs`.
/// `openfang-runtime` re-exports this type; all existing match arms compile
/// unchanged.
#[derive(Error, Debug)]
pub enum LlmError {
    /// HTTP request failed.
    #[error("HTTP error: {0}")]
    Http(String),

    /// The API returned a non-2xx status.
    #[error("API error ({status}): {message}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Error message from the API.
        message: String,
    },

    /// Rate-limited — caller should retry after the given delay.
    #[error("Rate limited, retry after {retry_after_ms}ms")]
    RateLimited {
        /// Milliseconds to wait before retrying.
        retry_after_ms: u64,
    },

    /// Response JSON could not be parsed.
    #[error("Parse error: {0}")]
    Parse(String),

    /// No API key configured for the provider.
    #[error("Missing API key: {0}")]
    MissingApiKey(String),

    /// Model overloaded by the provider.
    #[error("Model overloaded, retry after {retry_after_ms}ms")]
    Overloaded {
        /// Milliseconds to wait before retrying.
        retry_after_ms: u64,
    },

    /// Authentication failed (invalid or revoked API key).
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    /// The requested model is not available on this provider.
    #[error("Model not found: {0}")]
    ModelNotFound(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// LlmDriver — minimal interface (complete() only)
// ─────────────────────────────────────────────────────────────────────────────

/// Core LLM driver trait — non-streaming interface only.
///
/// Only `complete()` is defined here because `stream()` requires
/// `tokio::sync::mpsc::Sender<StreamEvent>` which would pull `tokio` and
/// `StreamEvent` (a UX-rich type) into `openfang-types`.
///
/// The streaming extension lives in `openfang-runtime::llm_driver::LlmDriverExt`
/// and is blanket-implemented for every type that implements this trait.
///
/// ## How `openfang-runtime` re-exports this
///
/// ```rust
/// // openfang-runtime/src/llm_driver.rs
/// pub use openfang_types::driver::{
///     CompletionRequest, CompletionResponse, LlmDriver, LlmError,
/// };
///
/// // Remove (or keep as type alias) the local definitions of these types.
/// ```
///
/// All existing code (`kernel.rs`, drivers, tests) that imports from
/// `openfang_runtime::llm_driver` continues to compile unchanged.
#[async_trait]
pub trait LlmDriver: Send + Sync {
    /// Send a completion request and wait for the full response.
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, LlmError>;

    /// Stream a completion request, sending incremental [`StreamEvent`]s to `tx`.
    ///
    /// The default implementation wraps `complete()` — providers with native
    /// SSE streaming (Anthropic, OpenAI, Gemini, …) override this method in
    /// their `impl LlmDriver for X` block.
    ///
    /// ## Why stream() lives here (not in a separate LlmDriverExt trait)
    ///
    /// `dyn LlmDriver` trait objects (stored as `Arc<dyn LlmDriver>` throughout
    /// the kernel and agent loop) can only dispatch to methods that are in the
    /// trait's vtable.  Putting `stream()` in a separate `LlmDriverExt` trait
    /// prevents dynamic dispatch from reaching specific driver implementations.
    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        let response = self.complete(request).await?;
        let text = response.text();
        if !text.is_empty() {
            let _ = tx.send(StreamEvent::TextDelta { text }).await;
        }
        let _ = tx
            .send(StreamEvent::ContentComplete {
                stop_reason: response.stop_reason.clone(),
                usage: response.usage.clone(),
            })
            .await;
        Ok(response)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding driver  (sub-module so callers import selectively)
// ─────────────────────────────────────────────────────────────────────────────

/// Embedding driver trait and error type.
///
/// The trait only uses `&str` and `Vec<f32>` so it has no external dependencies
/// beyond `async_trait`.  Concrete implementations (`OpenAIEmbeddingDriver`,
/// FastEmbed, etc.) stay in `openfang-runtime` where `reqwest` and `zeroize`
/// are already available.
///
/// ## How `openfang-runtime` re-exports this
///
/// ```rust
/// // openfang-runtime/src/embedding.rs
/// pub use openfang_types::driver::embedding::{EmbeddingDriver, EmbeddingError};
///
/// // Keep OpenAIEmbeddingDriver, create_embedding_driver, cosine_similarity, etc. here.
/// ```
pub mod embedding {
    use async_trait::async_trait;
    use thiserror::Error;

    /// Errors that can occur when computing embeddings.
    ///
    /// **Direct equivalent** of `EmbeddingError` in
    /// `openfang-runtime/src/embedding.rs`.
    #[derive(Debug, Error)]
    pub enum EmbeddingError {
        /// HTTP request failed.
        #[error("HTTP error: {0}")]
        Http(String),

        /// The API returned a non-2xx status.
        #[error("API error (status {status}): {message}")]
        Api {
            /// HTTP status code.
            status: u16,
            /// Error message from the API.
            message: String,
        },

        /// Response JSON could not be parsed.
        #[error("Parse error: {0}")]
        Parse(String),

        /// No API key configured for the provider.
        #[error("Missing API key: {0}")]
        MissingApiKey(String),
    }

    /// Core embedding driver trait.
    ///
    /// Implemented by `OpenAIEmbeddingDriver` (and future local/remote drivers)
    /// in `openfang-runtime`.  Only primitive types are used here so the trait
    /// can live in `openfang-types` without pulling in `reqwest` or `zeroize`.
    #[async_trait]
    pub trait EmbeddingDriver: Send + Sync {
        /// Compute embedding vectors for a batch of texts.
        ///
        /// Returns one `Vec<f32>` per input text, in the same order.
        async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

        /// Compute the embedding for a single text string.
        ///
        /// Defaults to calling `embed(&[text])` and extracting the first result;
        /// implementations may override with a more efficient single-item call.
        async fn embed_one(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
            self.embed(&[text])
                .await?
                .into_iter()
                .next()
                .ok_or_else(|| EmbeddingError::Parse("Empty embedding response".to_string()))
        }

        /// Return the dimensionality of embeddings produced by this driver.
        ///
        /// Used by the semantic store to size BLOB columns and validate
        /// that stored and queried embeddings have compatible shapes.
        fn dimensions(&self) -> usize;
    }
}