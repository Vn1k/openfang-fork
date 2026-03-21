//! LLM-based fact extraction — core of mem0-style memory.
//!
//! Converts raw conversation messages into discrete, atomic facts
//! that can be independently stored, updated, or deleted.
//!
//! mem0 approach:
//!   1. Parse messages into a text blob
//!   2. LLM extracts a list of atomic facts (JSON: {"facts": ["...", "..."]})
//!   3. Facts are returned for downstream consolidation
//!
//! ## Why `openfang_types::driver` instead of `openfang_runtime`
//!
//! `openfang-runtime` already depends on `openfang-memory`.  Importing back
//! from `openfang-runtime` here would create a circular dependency.  The
//! driver traits (`LlmDriver`, `CompletionRequest`, …) now live in the leaf
//! crate `openfang-types::driver` so both crates can reference them safely.
//! `openfang-runtime` re-exports these types so all existing callers
//! (e.g. `kernel.rs`) continue to compile unchanged.

use crate::prompt as prompts;
use openfang_types::driver::{CompletionRequest, LlmDriver};
use openfang_types::error::{OpenFangError, OpenFangResult};
use openfang_types::message::Message;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single extracted fact from a conversation.
#[derive(Debug, Clone)]
pub struct ExtractedFact {
    pub content: String,
    /// Role of who produced this fact: `"user"` or `"assistant"`.
    pub role: Option<String>,
    /// Optional actor/agent identifier.
    pub actor_id: Option<String>,
}

/// Selects which side of the conversation to extract facts from.
pub enum ExtractionMode {
    /// Extract personal facts from **user messages only**.
    User,
    /// Extract facts about the **assistant** from assistant messages only.
    Agent,
}

// ─────────────────────────────────────────────────────────────────────────────
// Prompt builder
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the fully-resolved extraction system prompt for the given mode.
///
/// Delegates to `prompt.rs` so the canonical prompt text lives in one place.
pub fn build_extraction_prompt(mode: ExtractionMode) -> String {
    match mode {
        ExtractionMode::User => prompts::user_memory_extraction_prompt(),
        ExtractionMode::Agent => prompts::agent_memory_extraction_prompt(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main extraction entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Extract discrete facts from a list of messages using the LLM.
///
/// # Arguments
/// * `messages`        - Full conversation; system messages are skipped.
/// * `is_agent_memory` - `true` → use agent-focused prompt; `false` → user.
/// * `llm_driver`      - Driver used to call the LLM.
///
/// # Returns
/// `Vec<ExtractedFact>` — empty if the conversation carries nothing memorable.
pub async fn extract_facts(
    messages: &[Message],
    is_agent_memory: bool,
    llm_driver: &dyn LlmDriver,
    model: &str,
) -> OpenFangResult<Vec<ExtractedFact>> {
    use openfang_types::message::{MessageContent, Role};

    // Build plain-text conversation block (mirrors mem0's parse_messages).
    // System messages are intentionally skipped — they must not influence
    // fact extraction per mem0's USER_MEMORY_EXTRACTION_PROMPT rules.
    let mut conversation_text = String::new();
    for msg in messages {
        let label = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::System => continue,
        };
        let content = msg.content.text_content();
        if !content.is_empty() {
            conversation_text.push_str(&format!("{}: {}\n", label, content));
        }
    }

    if conversation_text.trim().is_empty() {
        return Ok(vec![]);
    }

    let system_prompt = build_extraction_prompt(if is_agent_memory {
        ExtractionMode::Agent
    } else {
        ExtractionMode::User
    });

    let request = CompletionRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Text(format!("Input:\n{}", conversation_text)),
        }],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.1, // Low temperature → consistent, deterministic JSON
        system: Some(system_prompt),
        thinking: None,
    };

    let response = llm_driver
        .complete(request)
        .await
        .map_err(|e| OpenFangError::LlmDriver(e.to_string()))?;

    let raw = response.text();

    let facts = parse_facts_json(&raw).unwrap_or_default();

    // Convert to ExtractedFact.
    // `role` / `actor_id` left None here; callers with richer context
    // (e.g. smart_memory.rs) can enrich them after extraction.
    let extracted = facts
        .into_iter()
        .map(|content| ExtractedFact {
            content,
            role: None,
            actor_id: None,
        })
        .collect();

    Ok(extracted)
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse the `{"facts": [...]}` JSON returned by the LLM.
///
/// Strips markdown code-fences if present, then deserialises the `facts`
/// array.  Returns `None` only on hard parse failure so callers can safely
/// call `.unwrap_or_default()`.
fn parse_facts_json(text: &str) -> Option<Vec<String>> {
    // Find outermost `{...}` block, tolerating ``` code-fence wrappers.
    let start = text.find('{')?;
    let end = text.rfind('}').map(|i| i + 1)?;

    if end <= start {
        return None;
    }

    let clean = &text[start..end];
    let value: serde_json::Value = serde_json::from_str(clean).ok()?;
    let facts = value.get("facts")?.as_array()?;

    Some(
        facts
            .iter()
            .filter_map(|f| f.as_str().map(String::from))
            .filter(|s| !s.is_empty())
            .collect(),
    )
}