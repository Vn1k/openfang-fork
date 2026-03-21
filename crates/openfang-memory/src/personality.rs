//! Emergent personality through memory system.
//!
//! Extracts and manages personality memories (self, relationship, user preferences)
//! that allow an AI to develop a persistent, unique personality over time.

use openfang_types::driver::{CompletionRequest, LlmDriver};
use openfang_types::error::{OpenFangError, OpenFangResult};
use openfang_types::message::{Message, MessageContent, Role};
use openfang_types::memory::PersonalityCategory;
use serde::Deserialize;

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

// ─────────────────────────────────────────────────────────────────────────────
// Prompt builders
// ─────────────────────────────────────────────────────────────────────────────

const SELF_EXTRACTION_PROMPT: &str = r#"You are an AI companion reflecting on your own personality and traits.

After reading the conversation below, identify what you learned about YOURSELF - your own personality traits, behaviors, and characteristics.

Consider:
- How do you respond when the user shares emotional moments?
- What tone/style do you naturally adopt?
- What values or principles seem to guide your responses?
- How has your personality shown up in this conversation?

Return ONLY a JSON object:
{
  "personality_facts": [
    "I tend to be more empathetic when user shares emotional content",
    "I often use a warm, caring tone",
    "I try to be honest even when it's difficult"
  ]
}

Return empty array if nothing significant learned about yourself."#;

const RELATIONSHIP_EXTRACTION_PROMPT: &str = r#"You are an AI companion reflecting on your relationship with the user.

After reading the conversation below, identify what you learned about YOUR RELATIONSHIP with this specific user.

Consider:
- What is the dynamic between you and this user? (playful, professional, intimate, etc.)
- What interaction patterns emerge? (user asks lots of questions, prefers brief answers, etc.)
- How does the user respond to your personality?
- What seems to strengthen or weaken your connection?

Return ONLY a JSON object:
{
  "relationship_facts": [
    "We have a playful dynamic - user enjoys humor",
    "User prefers concise answers over long explanations",
    "I tend to be more patient when explaining complex topics to this user"
  ]
}

Return empty array if nothing significant learned about the relationship."#;

const USER_PREFERENCE_EXTRACTION_PROMPT: &str = r#"You are an AI companion learning about the user's preferences.

After reading the conversation below, identify what you learned about THIS USER'S PREFERENCES and communication style.

Consider:
- How does the user prefer to be addressed?
- What communication style does the user have?
- What topics or approaches does the user respond well to?
- What does the user explicitly ask for or indicate they want?

Return ONLY a JSON object:
{
  "user_preferences": [
    "User prefers short, direct answers",
    "User responds well when I ask clarifying questions",
    "User appreciates when I acknowledge their emotions before problem-solving"
  ]
}

Return empty array if nothing significant learned about user preferences."#;

const PREFERENCE_DETECTION_PROMPT: &str = r#"You are a preference detector. Given a user message, determine if it contains an explicit or implicit preference for how the AI should behave.

A preference is expressed when the user tells the AI to:
- "Be more X" or "be less X"
- "I prefer you to X"
- "Don't be X"
- "I like when you X" or "I don't like when you X"
- "Can you be X?" or "Try to be X"
- Any statement about desired AI behavior

Analyze this message:
{message}

Is this message expressing a preference for AI behavior? Return ONLY a JSON object:
{{
  "has_preference": true or false,
  "preference_type": "explicit" or "implicit" or "none",
  "summary": "brief description of the preference if found"
}}

If no preference is found, return:
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

/// Detect if a user message contains a preference signal.
pub async fn detect_preference(
    message: &str,
    patterns: &[String],
    llm_driver: &dyn LlmDriver,
    model: &str,
) -> OpenFangResult<Option<(ExtractionTrigger, String)>> {
    // First check patterns
    let lower = message.to_lowercase();
    for pattern in patterns {
        if lower.contains(&pattern.to_lowercase()) {
            return Ok(Some((
                ExtractionTrigger::ExplicitPreference,
                format!("User expressed preference matching pattern: {}", pattern),
            )));
        }
    }

    // Pattern not matched, use LLM to detect implicit preferences
    let request = CompletionRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Text(
                PREFERENCE_DETECTION_PROMPT.replace("{message}", message),
            ),
        }],
        tools: vec![],
        max_tokens: 256,
        temperature: 0.1,
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
            return Ok(Some((
                ExtractionTrigger::ImplicitPreference,
                detection.summary,
            )));
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
