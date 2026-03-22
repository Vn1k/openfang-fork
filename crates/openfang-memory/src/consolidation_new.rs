//! LLM-based memory consolidation — the mem0 "smart merge" pipeline.
//!
//! After extracting new facts, this module:
//!   1. Searches existing memories for similar content
//!   2. Passes old + new memories to LLM to decide action
//!   3. Returns ADD / UPDATE / DELETE / NONE decisions (caller executes them)

use crate::extraction::ExtractedFact;
use crate::prompt as prompts;
use crate::MemorySubstrate;
use openfang_types::agent::AgentId;
use openfang_types::driver::embedding::EmbeddingDriver;
use openfang_types::driver::{CompletionRequest, LlmDriver};
use openfang_types::error::{OpenFangError, OpenFangResult};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Action decided by the LLM for one memory entry.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAction {
    Add { text: String },
    Update { memory_id: String, text: String, old_text: String },
    Delete { memory_id: String, text: String },
    None,
}

/// Output of one full consolidation run.
#[derive(Debug)]
pub struct ConsolidationResult {
    pub actions: Vec<MemoryAction>,
    pub facts_processed: usize,
    pub existing_checked: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Consolidation prompt
//
// Re-uses DEFAULT_UPDATE_MEMORY_PROMPT from prompt.rs — single source of truth.
// The prompt is assembled via update_memory_messages() which mirrors
// mem0's get_update_memory_messages() including the "memory is empty" branch.
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Main consolidation entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full mem0-style consolidation pipeline.
///
/// ## EmbeddingDriver bound
///
/// The parameter uses `dyn EmbeddingDriver + Send + Sync` (not just `dyn EmbeddingDriver`)
/// to match what `kernel.rs` passes: `Arc<dyn EmbeddingDriver + Send + Sync>.as_deref()`
/// produces `Option<&(dyn EmbeddingDriver + Send + Sync)>`.
pub async fn consolidate_memories(
    agent_id: AgentId,
    new_facts: Vec<ExtractedFact>,
    memory: &MemorySubstrate,
    embedding_driver: Option<&(dyn EmbeddingDriver + Send + Sync)>,
    llm_driver: &dyn LlmDriver,
    model: &str,
) -> OpenFangResult<ConsolidationResult> {
    use openfang_types::memory::MemoryFilter;

    if new_facts.is_empty() {
        return Ok(ConsolidationResult {
            actions: vec![],
            facts_processed: 0,
            existing_checked: 0,
        });
    }

    let filter = MemoryFilter {
        agent_id: Some(agent_id),
        scope: Some("semantic".to_string()),
        ..Default::default()
    };

    // ── Search similar existing memories for each new fact ────────────────────
    let mut all_existing: HashMap<String, String> = HashMap::new();

    for fact in &new_facts {
        let results = match embedding_driver {
            Some(emb) => match emb.embed_one(&fact.content).await {
                Ok(vec) => memory.recall_with_embedding(
                    &fact.content, 5, Some(filter.clone()), Some(&vec),
                )?,
                Err(_) => memory.recall_with_embedding(
                    &fact.content, 5, Some(filter.clone()), None,
                )?,
            },
            None => memory.recall_with_embedding(
                &fact.content, 5, Some(filter.clone()), None,
            )?,
        };

        for mem in results {
            all_existing.entry(mem.id.0.to_string()).or_insert(mem.content);
        }
    }

    let existing_count = all_existing.len();

    // ── Fast path: no existing memories → every fact is an ADD ───────────────
    if all_existing.is_empty() {
        let actions = new_facts.iter()
            .map(|f| MemoryAction::Add { text: f.content.clone() })
            .collect();
        return Ok(ConsolidationResult {
            actions,
            facts_processed: new_facts.len(),
            existing_checked: 0,
        });
    }

    // ── Map UUIDs → sequential integers (prevents LLM hallucination) ─────────
    let id_list: Vec<String> = all_existing.keys().cloned().collect();
    let int_to_uuid: HashMap<String, String> = id_list.iter().enumerate()
        .map(|(i, uuid)| (i.to_string(), uuid.clone()))
        .collect();

    // ── Build prompt via prompt.rs (single source of truth) ─────────────────
    // Format existing memories as JSON array matching mem0's convention
    let existing_json: Vec<serde_json::Value> = id_list.iter().enumerate()
        .map(|(i, uuid)| serde_json::json!({
            "id": i.to_string(),
            "text": all_existing[uuid]
        }))
        .collect();
    let existing_str = serde_json::to_string(&existing_json).unwrap_or_default();

    let new_facts_str: String = new_facts.iter()
        .map(|f| f.content.clone())
        .collect::<Vec<_>>()
        .join("\n");

    // update_memory_messages() mirrors mem0's get_update_memory_messages()
    // Includes: DEFAULT_UPDATE_MEMORY_PROMPT + existing memory block + new facts
    let prompt = prompts::update_memory_messages(
        Some(&existing_str),
        &new_facts_str,
        None, // use DEFAULT_UPDATE_MEMORY_PROMPT
    );

    // ── Call LLM ─────────────────────────────────────────────────────────────
    let request = CompletionRequest {
        model: model.to_string(),
        messages: vec![openfang_types::message::Message {
            role: openfang_types::message::Role::User,
            content: openfang_types::message::MessageContent::Text(prompt),
        }],
        tools: vec![],
        max_tokens: 2048,
        temperature: 0.1,
        system: None,
        thinking: None,
    };

    let response = llm_driver
        .complete(request)
        .await
        .map_err(|e| OpenFangError::LlmDriver(e.to_string()))?;

    // Fix: bind to a local so &str lifetime is valid
    let response_text = response.text();
    let actions = parse_memory_actions(&response_text, &int_to_uuid)?;

    Ok(ConsolidationResult {
        actions,
        facts_processed: new_facts.len(),
        existing_checked: existing_count,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Response parser
// ─────────────────────────────────────────────────────────────────────────────

fn parse_memory_actions(
    text: &str,
    int_to_uuid: &HashMap<String, String>,
) -> OpenFangResult<Vec<MemoryAction>> {
    let start = text.find('{').ok_or_else(|| {
        OpenFangError::Serialization("No JSON object in consolidation response".to_string())
    })?;
    let end = text.rfind('}').map(|i| i + 1).ok_or_else(|| {
        OpenFangError::Serialization("Malformed JSON in consolidation response".to_string())
    })?;
    if end <= start {
        return Err(OpenFangError::Serialization("Empty JSON object".to_string()));
    }

    let value: serde_json::Value = serde_json::from_str(&text[start..end])
        .map_err(|e| OpenFangError::Serialization(e.to_string()))?;

    let memory_array = value.get("memory").and_then(|v| v.as_array())
        .ok_or_else(|| OpenFangError::Serialization("No 'memory' array".to_string()))?;

    let mut actions = Vec::new();
    for item in memory_array {
        let event = item.get("event").and_then(|v| v.as_str()).unwrap_or("NONE");
        let text = item.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let int_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();

        match event {
            "ADD" => {
                if !text.is_empty() { actions.push(MemoryAction::Add { text }); }
            }
            "UPDATE" => {
                if let Some(real_uuid) = int_to_uuid.get(&int_id) {
                    let old_text = item.get("old_memory").and_then(|v| v.as_str())
                        .unwrap_or("").to_string();
                    actions.push(MemoryAction::Update {
                        memory_id: real_uuid.clone(), text, old_text,
                    });
                }
            }
            "DELETE" => {
                if let Some(real_uuid) = int_to_uuid.get(&int_id) {
                    actions.push(MemoryAction::Delete { memory_id: real_uuid.clone(), text });
                }
            }
            _ => {}
        }
    }
    Ok(actions)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn parses_add() {
        let json = r#"{"memory":[{"id":null,"text":"Loves Rust","event":"ADD"}]}"#;
        let actions = parse_memory_actions(json, &map(&[])).unwrap();
        assert!(matches!(&actions[0], MemoryAction::Add { text } if text == "Loves Rust"));
    }

    #[test]
    fn parses_update_with_id_resolution() {
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let json = r#"{"memory":[{"id":"0","text":"Loves Rust and Python","event":"UPDATE","old_memory":"Loves Rust"}]}"#;
        let actions = parse_memory_actions(json, &map(&[("0", uuid)])).unwrap();
        match &actions[0] {
            MemoryAction::Update { memory_id, text, old_text } => {
                assert_eq!(memory_id, uuid);
                assert_eq!(text, "Loves Rust and Python");
                assert_eq!(old_text, "Loves Rust");
            }
            _ => panic!("Expected Update"),
        }
    }

    #[test]
    fn drops_none_and_unknown_ids() {
        let json = r#"{"memory":[{"id":"99","text":"ghost","event":"UPDATE","old_memory":"x"},{"id":"0","text":"y","event":"NONE"}]}"#;
        assert!(parse_memory_actions(json, &map(&[("0", "uuid")])).unwrap().is_empty());
    }

    #[test]
    fn errors_on_invalid_json() {
        assert!(parse_memory_actions("not json", &map(&[])).is_err());
    }
}