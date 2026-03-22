//! Emergent personality through memory system.
//!
//! Extracts and manages personality memories (self, relationship, user preferences)
//! that allow an AI to develop a persistent, unique personality over time.

use crate::prompt as prompts;
use openfang_types::driver::{CompletionRequest, LlmDriver};
use openfang_types::error::{OpenFangError, OpenFangResult};
use openfang_types::message::{Message, MessageContent, Role};
use openfang_types::memory::PersonalityCategory;
use serde::Deserialize;
use std::collections::HashMap;

/// A single extracted personality memory.
#[derive(Debug, Clone)]
pub struct ExtractedPersonality {
    pub content: String,
    pub category: PersonalityCategory,
    pub locked: bool,
}

/// Trigger type for personality extraction.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtractionTrigger {
    /// Periodic trigger (every N conversations).
    Periodic,
    /// User expressed an explicit preference.
    ExplicitPreference,
    /// LLM detected implicit preference signal.
    ImplicitPreference,
}

const PREFERENCE_DETECTION_PROMPT: &str = r#"You are analyzing recent conversation messages to detect whether the user is signaling — explicitly or implicitly — how they want the AI to behave.

EXPLICIT signals (user directly states a preference):
- "bisa lebih singkat?" / "can you be more concise?"
- "skip the explanation, just give me the answer"
- "i prefer bullet points"
- "don't be so formal"
- "stop asking me questions, just do it"

IMPLICIT signals (behavioral cues that reveal preference without stating it):
- User repeatedly interrupts or ignores long answers → wants brevity
- User says "ok ok i get it" or "yeah yeah" mid-explanation → over-explained
- User gives short replies to long responses → mismatch in communication style
- User says "that's not what i meant" → AI misunderstood intent
- User asks the same question differently → previous answer was unsatisfying
- User expresses frustration or impatience → something is not working
- User says "finally!" or "exactly!" → something clicked after resistance
- User asks AI to redo something → quality or style was off

Analyze these recent messages:
{messages}

Does this exchange contain a preference signal (explicit or implicit)?

Return ONLY valid JSON. No explanation, no markdown, no preamble:
{{"has_preference": true, "preference_type": "explicit", "summary": "user wants shorter responses"}}
or
{{"has_preference": false, "preference_type": "none", "summary": ""}}"#;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Extract personality memories from a conversation.
///
/// # Arguments
/// * `messages`     - Full conversation history.
/// * `trigger`      - What triggered this extraction.
/// * `llm_driver`  - LLM driver for extraction.
/// * `model`        - Model to use for extraction.
///
/// # Returns
/// `Vec<ExtractedPersonality>` — personality memories categorized.
/// Extract personality observations using a single unified LLM call.
///
/// Replaces the previous 3-call approach (self + relationship + user_preference).
/// A single LLM now extracts AND categorizes all observations together, which:
/// - Eliminates cross-category confusion (facts about user landing in "self")
/// - Reduces LLM calls: extraction 3 → 1 (total pipeline: 5 → 2)
/// - Mirrors smart_memory's proven single-call extraction approach
pub async fn extract_personality(
    messages: &[Message],
    _trigger: ExtractionTrigger,
    llm_driver: &dyn LlmDriver,
    model: &str,
) -> OpenFangResult<Vec<ExtractedPersonality>> {
    let conversation_text = build_conversation_text(messages);
    if conversation_text.trim().is_empty() {
        return Ok(vec![]);
    }

    let request = CompletionRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Text(format!("Conversation:\n{}", conversation_text)),
        }],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.1,
        system: Some(prompts::PERSONALITY_EXTRACTION_PROMPT.to_string()),
        thinking: None,
    };

    let response = llm_driver
        .complete(request)
        .await
        .map_err(|e| OpenFangError::LlmDriver(e.to_string()))?;

    let raw = response.text();
    parse_unified_personality_json(&raw)
}

fn parse_unified_personality_json(text: &str) -> OpenFangResult<Vec<ExtractedPersonality>> {
    let start = text.find('{').ok_or_else(|| {
        OpenFangError::Serialization("No JSON in unified personality response".to_string())
    })?;
    let end = text.rfind('}').map(|i| i + 1).ok_or_else(|| {
        OpenFangError::Serialization("Malformed JSON in unified personality response".to_string())
    })?;
    if end <= start {
        return Ok(vec![]);
    }

    let value: serde_json::Value = serde_json::from_str(&text[start..end])
        .map_err(|e| OpenFangError::Serialization(e.to_string()))?;

    let observations = match value.get("observations").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return Ok(vec![]),
    };

    let mut result = Vec::new();
    for obs in observations {
        let content = match obs.get("content").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => continue,
        };
        let category_str = obs.get("category").and_then(|v| v.as_str()).unwrap_or("self");

        let (category, locked) = match category_str {
            "self" => (PersonalityCategory::Self_, true),
            "relationship" => (PersonalityCategory::Relationship, false),
            "user_preference" => (PersonalityCategory::UserPreference, false),
            _ => continue,
        };

        result.push(ExtractedPersonality { content, category, locked });
    }

    Ok(result)
}


/// Action decided by consolidation LLM for one personality memory entry.
#[derive(Debug, Clone, PartialEq)]
pub enum PersonalityAction {
    Add { text: String, category: PersonalityCategory, locked: bool },
    Update { memory_id: String, text: String, old_text: String },
    Delete { memory_id: String },
    None,
}

/// Result of one full personality consolidation run.
#[derive(Debug)]
pub struct PersonalityConsolidationResult {
    pub actions: Vec<PersonalityAction>,
    pub facts_processed: usize,
    pub existing_checked: usize,
}

/// A reference to an existing personality memory for consolidation input.
#[derive(Debug, Clone)]
pub struct ExistingPersonalityMemory {
    pub id: String,
    pub content: String,
    pub locked: bool,
}

/// Consolidate newly extracted personality facts against existing memories.
///
/// Passes all existing memories to LLM for holistic deduplication — personality
/// memories are few enough that loading all is more accurate than per-fact recall.
/// Locked (Self_) memories are protected from deletion.
pub async fn consolidate_personality(
    new_facts: Vec<ExtractedPersonality>,
    existing: Vec<ExistingPersonalityMemory>,
    llm_driver: &dyn LlmDriver,
    model: &str,
) -> OpenFangResult<PersonalityConsolidationResult> {
    if new_facts.is_empty() {
        return Ok(PersonalityConsolidationResult {
            actions: vec![],
            facts_processed: 0,
            existing_checked: 0,
        });
    }

    // Fast path: no existing memories → all new facts are ADD
    if existing.is_empty() {
        let actions = new_facts.iter().map(|f| PersonalityAction::Add {
            text: f.content.clone(),
            category: f.category.clone(),
            locked: f.locked,
        }).collect();
        return Ok(PersonalityConsolidationResult {
            actions,
            facts_processed: new_facts.len(),
            existing_checked: 0,
        });
    }

    let existing_count = existing.len();

    // Map sequential int ID → real UUID (prevents LLM hallucinating new IDs)
    let int_to_uuid: HashMap<String, String> = existing.iter().enumerate()
        .map(|(i, m)| (i.to_string(), m.id.clone()))
        .collect();
    let uuid_to_locked: HashMap<String, bool> = existing.iter()
        .map(|m| (m.id.clone(), m.locked))
        .collect();

    // Build existing memories string with locked marker
    let existing_str: String = existing.iter().enumerate()
        .map(|(i, m)| {
            let lock_marker = if m.locked { " [locked]" } else { "" };
            format!("ID {}{}: {}", i, lock_marker, m.content)
        })
        .collect::<Vec<_>>()
        .join("\n");

    let new_facts_str: String = new_facts.iter()
        .map(|f| format!("- {}", f.content))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt_text = prompts::PERSONALITY_CONSOLIDATION_PROMPT
        .replace("{existing_memories}", &existing_str)
        .replace("{new_facts}", &new_facts_str);

    let request = CompletionRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Text(prompt_text),
        }],
        tools: vec![],
        max_tokens: 4096,
        temperature: 0.1,
        system: None,
        thinking: None,
    };

    let response = llm_driver
        .complete(request)
        .await
        .map_err(|e| OpenFangError::LlmDriver(e.to_string()))?;

    let raw = response.text();
    let actions = parse_personality_actions(
        &raw,
        &int_to_uuid,
        &uuid_to_locked,
        &new_facts,
    )?;

    Ok(PersonalityConsolidationResult {
        actions,
        facts_processed: new_facts.len(),
        existing_checked: existing_count,
    })
}

fn parse_personality_actions(
    text: &str,
    int_to_uuid: &HashMap<String, String>,
    uuid_to_locked: &HashMap<String, bool>,
    new_facts: &[ExtractedPersonality],
) -> OpenFangResult<Vec<PersonalityAction>> {
    let start = text.find('{').ok_or_else(|| {
        OpenFangError::Serialization("No JSON in personality consolidation response".to_string())
    })?;
    let end = text.rfind('}').map(|i| i + 1).ok_or_else(|| {
        OpenFangError::Serialization("Malformed JSON in personality consolidation response".to_string())
    })?;
    if end <= start {
        return Ok(vec![]);
    }

    let value: serde_json::Value = serde_json::from_str(&text[start..end])
        .map_err(|e| OpenFangError::Serialization(e.to_string()))?;

    let memory_array = value.get("memory").and_then(|v| v.as_array())
        .ok_or_else(|| OpenFangError::Serialization("No 'memory' array in response".to_string()))?;

    let mut actions = Vec::new();

    for item in memory_array {
        let event = item.get("event").and_then(|v| v.as_str()).unwrap_or("NONE");
        let text_val = item.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let int_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();

        match event {
            "ADD" => {
                if !text_val.is_empty() {
                    let matching_fact = new_facts.iter()
                        .find(|f| f.content.trim() == text_val.trim())
                        .cloned()
                        .or_else(|| new_facts.first().cloned());

                    if let Some(fact) = matching_fact {
                        actions.push(PersonalityAction::Add {
                            text: text_val,
                            category: fact.category,
                            locked: fact.locked,
                        });
                    }
                }
            }
            "UPDATE" => {
                if let Some(real_uuid) = int_to_uuid.get(&int_id) {
                    let old_text = item.get("old_memory")
                        .and_then(|v| v.as_str())
                        .unwrap_or("").to_string();
                    actions.push(PersonalityAction::Update {
                        memory_id: real_uuid.clone(),
                        text: text_val,
                        old_text,
                    });
                }
            }
            "DELETE" => {
                if let Some(real_uuid) = int_to_uuid.get(&int_id) {
                    let is_locked = uuid_to_locked.get(real_uuid).copied().unwrap_or(false);
                    if !is_locked {
                        actions.push(PersonalityAction::Delete {
                            memory_id: real_uuid.clone(),
                        });
                    }
                }
            }
            _ => {}
        }
    }

    Ok(actions)
}

/// Detect if recent messages contain a preference signal — explicit or implicit.
///
/// Analyzes the last N messages as a unit rather than a single message,
/// enabling detection of implicit behavioral signals spread across turns.
/// Detect if recent messages contain a preference signal — explicit or implicit.
///
/// Analyzes the last N messages as a unit rather than a single message,
/// enabling detection of implicit behavioral signals spread across turns.
/// No keyword gate — always uses LLM for consistent, context-aware detection.
pub async fn detect_preference(
    messages: &[Message],
    llm_driver: &dyn LlmDriver,
    model: &str,
) -> OpenFangResult<Option<(ExtractionTrigger, String)>> {
    // Build a compact text of the last 6 messages (3 exchanges)
    // Enough context to catch implicit signals without overloading the prompt
    let recent_text = build_recent_messages_text(messages, 6);
    if recent_text.trim().is_empty() {
        return Ok(None);
    }

    let request = CompletionRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Text(
                PREFERENCE_DETECTION_PROMPT.replace("{messages}", &recent_text),
            ),
        }],
        tools: vec![],
        max_tokens: 256,
        temperature: 0.0, // deterministic — this is a classification task
        system: None,
        thinking: None,
    };

    let response = llm_driver
        .complete(request)
        .await
        .map_err(|e| OpenFangError::LlmDriver(e.to_string()))?;

    let raw = response.text();
    if let Some(detection) = parse_preference_detection(&raw) {
        if detection.has_preference {
            let trigger = match detection.preference_type.as_str() {
                "explicit" => ExtractionTrigger::ExplicitPreference,
                _ => ExtractionTrigger::ImplicitPreference,
            };
            return Ok(Some((trigger, detection.summary)));
        }
    }

    Ok(None)
}

/// Check if periodic extraction should run based on conversation count.
pub fn should_extract_periodic(conversation_count: usize, interval: usize) -> bool {
    if interval == 0 || conversation_count == 0 {
        return false;
    }
    conversation_count % interval == 0
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

fn build_conversation_text(messages: &[Message]) -> String {
    let mut text = String::new();
    for msg in messages {
        let label = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::System => continue,
        };
        let content = msg.content.text_content();
        if !content.is_empty() {
            text.push_str(&format!("{}: {}\n", label, content));
        }
    }
    text
}

fn build_recent_messages_text(messages: &[Message], n: usize) -> String {
    let recent: Vec<_> = messages
        .iter()
        .rev()
        .filter(|m| m.role != Role::System)
        .take(n)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    let mut text = String::new();
    for msg in recent {
        let label = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::System => continue,
        };
        let content = msg.content.text_content();
        if !content.is_empty() {
            text.push_str(&format!("{}: {}\n", label, content));
        }
    }
    text
}

fn parse_preference_detection(text: &str) -> Option<PreferenceDetection> {
    let start = text.find('{')?;
    let end = text.rfind('}').map(|i| i + 1)?;
    if end <= start {
        return None;
    }

    let clean = &text[start..end];
    serde_json::from_str(clean).ok()
}

#[derive(Debug, Deserialize)]
struct PreferenceDetection {
    has_preference: bool,
    #[serde(default)]
    preference_type: String,
    #[serde(default)]
    summary: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_extract_periodic() {
        assert!(should_extract_periodic(5, 5)); // 5 % 5 == 0
        assert!(!should_extract_periodic(3, 5)); // 3 % 5 != 0
        assert!(!should_extract_periodic(0, 5)); // edge case
        assert!(!should_extract_periodic(5, 0)); // interval 0 = disabled
    }

    #[test]
    fn test_build_conversation_text() {
        let messages = vec![
            Message {
                role: Role::User,
                content: MessageContent::Text("Hello!".to_string()),
            },
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("Hi there!".to_string()),
            },
            Message {
                role: Role::System,
                content: MessageContent::Text("System prompt".to_string()),
            },
        ];

        let text = build_conversation_text(&messages);
        assert!(text.contains("User: Hello!"));
        assert!(text.contains("Assistant: Hi there!"));
        assert!(!text.contains("System")); // System messages skipped
    }
}