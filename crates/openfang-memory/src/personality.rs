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

// ─────────────────────────────────────────────────────────────────────────────
// Prompt builders
// ─────────────────────────────────────────────────────────────────────────────

const SELF_EXTRACTION_PROMPT: &str = r#"You are analyzing a conversation to extract facts about HOW THE AI BEHAVED in this specific interaction.

Your task: identify concrete, observable patterns in the AI's behavior — not generic traits, not aspirations, not what the AI "tries" to do.

ONLY extract facts that are clearly evidenced by what actually happened in the conversation.
DO NOT extract vague traits like "I am helpful" or "I try to be honest" — these are assumed baselines, not meaningful observations.
DO NOT invent or infer things that are not directly shown.
Write each fact in first person from the AI's perspective.

Good examples (specific, evidenced):
- "I used humor to diffuse tension when the user expressed frustration"
- "I gave step-by-step breakdowns when the user asked technical questions"
- "I asked follow-up questions rather than assuming what the user meant"
- "I kept responses under 3 sentences when the user gave short replies"

Bad examples (too vague, not evidenced):
- "I am caring and empathetic"
- "I try to be helpful"
- "I value honesty"

Return ONLY valid JSON. No explanation, no markdown, no preamble:
{"personality_facts": ["fact 1", "fact 2"]}

Return {"personality_facts": []} if nothing concrete and specific was observed."#;

const RELATIONSHIP_EXTRACTION_PROMPT: &str = r#"You are analyzing a conversation to extract facts about the DYNAMIC AND PATTERN between the AI and this specific user.

Your task: identify observable relationship patterns — how this particular AI-user pair interacts, not generic observations.

Focus on:
- The energy and tone of their interaction (casual vs formal, warm vs transactional, collaborative vs directive)
- How the user engages with the AI (trusting, skeptical, exploratory, task-focused)
- Recurring interaction patterns that define THIS relationship specifically
- What seems to work well or poorly between them

ONLY extract facts clearly supported by the conversation.
DO NOT extract what either party "should" do or generic advice.
Each fact should describe the relationship as it IS, not as it could be.

Good examples:
- "User treats AI as a thinking partner, often asking for opinions rather than just facts"
- "User skips pleasantries and goes straight to the point — interaction is purely task-focused"
- "User pushes back when they disagree, and the conversation becomes more collaborative after"
- "User frequently uses humor, and AI mirroring this creates visible rapport"

Bad examples:
- "We have a good relationship"
- "User seems to like talking to me"
- "I should be more patient with this user"

Return ONLY valid JSON. No explanation, no markdown, no preamble:
{"relationship_facts": ["fact 1", "fact 2"]}

Return {"relationship_facts": []} if no clear pattern is observable yet."#;

const USER_PREFERENCE_EXTRACTION_PROMPT: &str = r#"You are analyzing a conversation to extract THIS USER'S communication preferences and behavioral signals.

Your task: identify how this specific user prefers to interact — both from explicit statements AND implicit behavioral signals.

Implicit signals to look for:
- User cuts off long responses → prefers brevity
- User asks for more detail → prefers depth
- User uses informal language → prefers casual tone
- User ignores emotional acknowledgments → prefers direct problem-solving
- User asks follow-up questions → engaged and exploratory
- User gives one-word answers → busy, disengaged, or overwhelmed
- User circles back to the same topic → something important was missed or unresolved
- User corrects the AI → values precision over agreeableness

Explicit signals: anything the user directly states about how they want the AI to respond.

ONLY extract preferences clearly evidenced by the conversation.
Write each fact as a specific, actionable observation about the user.
DO NOT write generic traits like "user is smart" or "user likes AI".

Good examples:
- "User prefers bullet points over paragraphs when receiving instructions"
- "User gets impatient with caveats — wants the direct answer first"
- "User explicitly asked to skip explanations and just give the result"
- "User responds more positively when AI acknowledges the difficulty of their situation before solving"
- "User uses Indonesian when relaxed, switches to English for technical topics"

Bad examples:
- "User likes good answers"
- "User prefers helpful responses"
- "User is technical"

Return ONLY valid JSON. No explanation, no markdown, no preamble:
{"user_preferences": ["fact 1", "fact 2"]}

Return {"user_preferences": []} if no clear preference signals were observed."#;

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

    let mut all_facts: Vec<ExtractedPersonality> = Vec::new();

    // Extract self facts
    let self_facts = extract_category(
        &conversation_text,
        "self",
        SELF_EXTRACTION_PROMPT,
        llm_driver,
        model,
    )
    .await?;
    all_facts.extend(self_facts);

    // Extract relationship facts
    let relationship_facts = extract_category(
        &conversation_text,
        "relationship",
        RELATIONSHIP_EXTRACTION_PROMPT,
        llm_driver,
        model,
    )
    .await?;
    all_facts.extend(relationship_facts);

    // Extract user preference facts
    let user_facts = extract_category(
        &conversation_text,
        "user_preference",
        USER_PREFERENCE_EXTRACTION_PROMPT,
        llm_driver,
        model,
    )
    .await?;
    all_facts.extend(user_facts);

    Ok(all_facts)
}

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


/// Consolidate newly extracted personality facts against existing memories.
///
/// Unlike smart memory consolidation which uses embedding similarity for recall,
/// personality consolidation passes ALL existing memories in the same category —
/// personality memories are few (typically < 30 per category) and must be
/// compared holistically to avoid semantic duplicates.
///
/// ## Locked memory protection
/// Self_ memories (locked=true) are passed with a locked marker so the LLM
/// knows not to DELETE them — only ADD or UPDATE is allowed.
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

    // Build new facts string
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
        max_tokens: 2048,
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
                    // Match back to original extracted fact to get category + locked
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
                    // Enforce locked protection: never delete locked memories
                    let is_locked = uuid_to_locked.get(real_uuid).copied().unwrap_or(false);
                    if !is_locked {
                        actions.push(PersonalityAction::Delete {
                            memory_id: real_uuid.clone(),
                        });
                    }
                    // If locked, silently ignore DELETE — the LLM should not have issued it
                }
            }
            _ => {} // NONE — no action
        }
    }

    Ok(actions)
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

async fn extract_category(
    conversation: &str,
    category: &str,
    prompt: &str,
    llm_driver: &dyn LlmDriver,
    model: &str,
) -> OpenFangResult<Vec<ExtractedPersonality>> {
    let request = CompletionRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Text(format!("Conversation:\n{}", conversation)),
        }],
        tools: vec![],
        max_tokens: 1024,
        temperature: 0.1,
        system: Some(prompt.to_string()),
        thinking: None,
    };

    let response = llm_driver
        .complete(request)
        .await
        .map_err(|e| OpenFangError::LlmDriver(e.to_string()))?;

    let raw = response.text();
    let facts = parse_personality_json(&raw, category)?;

    Ok(facts)
}

fn parse_personality_json(text: &str, category: &str) -> OpenFangResult<Vec<ExtractedPersonality>> {
    let start = text.find('{').ok_or_else(|| {
        OpenFangError::Serialization("No JSON object in personality response".to_string())
    })?;
    let end = text.rfind('}').map(|i| i + 1).ok_or_else(|| {
        OpenFangError::Serialization("Malformed JSON in personality response".to_string())
    })?;
    if end <= start {
        return Ok(vec![]);
    }

    let clean = &text[start..end];
    let value: serde_json::Value = serde_json::from_str(clean)
        .map_err(|e| OpenFangError::Serialization(e.to_string()))?;

    let key = match category {
        "self" => "personality_facts",
        "relationship" => "relationship_facts",
        "user_preference" => "user_preferences",
        _ => return Ok(vec![]),
    };

    let facts: &[serde_json::Value] = value.get(key).and_then(|v| v.as_array()).map_or(&[], |v| v.as_slice());

    let personality_category = match category {
        "self" => PersonalityCategory::Self_,
        "relationship" => PersonalityCategory::Relationship,
        "user_preference" => PersonalityCategory::UserPreference,
        _ => return Ok(vec![]),
    };

    // Self facts are locked by default, others are editable
    let locked = category == "self";

    let result: Vec<ExtractedPersonality> = facts
        .iter()
        .filter_map(|f| f.as_str().map(|s| s.to_string()))
        .filter(|s| !s.is_empty())
        .map(|content| ExtractedPersonality {
            content,
            category: personality_category.clone(),
            locked,
        })
        .collect();

    Ok(result)
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

    #[test]
    fn test_parse_personality_json_self() {
        let json = r#"{"personality_facts": ["I am caring", "I try to be honest"]}"#;
        let facts = parse_personality_json(json, "self").unwrap();
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].category, PersonalityCategory::Self_);
        assert!(facts[0].locked); // Self facts are locked
    }

    #[test]
    fn test_parse_personality_json_relationship() {
        let json = r#"{"relationship_facts": ["We have a playful dynamic"]}"#;
        let facts = parse_personality_json(json, "relationship").unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].category, PersonalityCategory::Relationship);
        assert!(!facts[0].locked); // Relationship facts are editable
    }

    #[test]
    fn test_parse_personality_json_user_preference() {
        let json = r#"{"user_preferences": ["User prefers short answers"]}"#;
        let facts = parse_personality_json(json, "user_preference").unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].category, PersonalityCategory::UserPreference);
        assert!(!facts[0].locked); // User preference facts are editable
    }
}