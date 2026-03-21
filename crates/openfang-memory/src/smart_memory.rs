//! SmartMemory — orchestrates the full mem0-style pipeline.
//!
//! Integrates extraction + consolidation into a single public API.
//! Called by `kernel.rs` after each agent turn as a background `tokio::spawn`.

use crate::consolidation_new::{consolidate_memories, MemoryAction};
use crate::extraction::extract_facts;
use crate::MemorySubstrate;
use openfang_types::agent::AgentId;
use openfang_types::driver::embedding::EmbeddingDriver;
use openfang_types::driver::LlmDriver;
use openfang_types::error::OpenFangResult;
// Memory trait must be in scope for .remember() and .forget() to resolve.
use openfang_types::memory::{Memory, MemoryId, MemorySource};
use openfang_types::message::Message;
use std::collections::HashMap;
use tracing::{info, warn};

// ─────────────────────────────────────────────────────────────────────────────
// Public result type
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct SmartAddResult {
    pub added: u32,
    pub updated: u32,
    pub deleted: u32,
    pub facts_extracted: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full mem0-style smart memory pipeline for a completed agent turn.
///
/// ## EmbeddingDriver bound
///
/// `embedding_driver` uses `dyn EmbeddingDriver + Send + Sync` to match
/// what `kernel.rs` provides via `Arc<dyn EmbeddingDriver + Send + Sync>.as_deref()`.
/// Using just `dyn EmbeddingDriver` would be a type mismatch (E0308).
pub async fn smart_add(
    agent_id: AgentId,
    messages: &[Message],
    memory: &MemorySubstrate,
    llm_driver: &dyn LlmDriver,
    embedding_driver: Option<&(dyn EmbeddingDriver + Send + Sync)>,
    model: &str,
    _has_assistant_messages: bool,
) -> OpenFangResult<SmartAddResult> {
    // ── Step 1: Extract facts ─────────────────────────────────────────────────
    let facts = extract_facts(messages, false, llm_driver).await?;

    if facts.is_empty() {
        info!(agent_id = %agent_id, "No facts extracted from messages");
        return Ok(SmartAddResult { added: 0, updated: 0, deleted: 0, facts_extracted: 0 });
    }

    info!(agent_id = %agent_id, facts = facts.len(), "Extracted facts from messages");

    // ── Step 2 + 3: Consolidate with existing memories ────────────────────────
    let result = consolidate_memories(
        agent_id,
        facts.clone(),
        memory,
        embedding_driver,
        llm_driver,
        model,
    )
    .await?;

    info!(
        agent_id = %agent_id,
        actions = result.actions.len(),
        existing_checked = result.existing_checked,
        "Memory consolidation complete"
    );

    // ── Step 4: Execute actions ───────────────────────────────────────────────
    let mut added = 0u32;
    let mut updated = 0u32;
    let mut deleted = 0u32;

    for action in result.actions {
        match action {
            // ── ADD ───────────────────────────────────────────────────────────
            MemoryAction::Add { text } => {
                let embedding = embed_opt(embedding_driver, &text).await;

                let store_result = if let Some(ref vec) = embedding {
                    // remember_with_embedding is a plain (sync) method on MemorySubstrate
                    memory.remember_with_embedding(
                        agent_id,
                        &text,
                        MemorySource::Conversation,
                        "semantic",
                        HashMap::new(),
                        Some(vec),
                    )
                } else {
                    // remember() is an async method from the Memory trait
                    memory.remember(
                        agent_id,
                        &text,
                        MemorySource::Conversation,
                        "semantic",
                        HashMap::new(),
                    )
                    .await
                };

                match store_result {
                    Ok(id) => {
                        added += 1;
                        log_history_error(
                            memory.add_memory_history(&id.0.to_string(), None, Some(&text), "ADD"),
                        );
                    }
                    Err(e) => warn!("smart_add: failed to add memory: {e}"),
                }
            }

            // ── UPDATE ────────────────────────────────────────────────────────
            MemoryAction::Update { memory_id, text, old_text } => {
                let mem_id = match parse_memory_id(&memory_id) {
                    Some(id) => id,
                    None => continue,
                };

                // update_embedding is the real method name on MemorySubstrate
                if let Some(vec) = embed_opt(embedding_driver, &text).await {
                    if let Err(e) = memory.update_embedding(mem_id, &vec) {
                        warn!("smart_add: failed to update embedding for {memory_id}: {e}");
                    }
                }

                // update_memory_content is a new method added via substrate_patch.rs
                match memory.update_memory_content(mem_id, &text) {
                    Ok(()) => {
                        updated += 1;
                        log_history_error(memory.add_memory_history(
                            &memory_id, Some(&old_text), Some(&text), "UPDATE",
                        ));
                    }
                    Err(e) => warn!("smart_add: failed to update memory {memory_id}: {e}"),
                }
            }

            // ── DELETE ────────────────────────────────────────────────────────
            MemoryAction::Delete { memory_id, text } => {
                let mem_id = match parse_memory_id(&memory_id) {
                    Some(id) => id,
                    None => continue,
                };

                // forget() is async — from the Memory trait
                match memory.forget(mem_id).await {
                    Ok(()) => {
                        deleted += 1;
                        log_history_error(memory.add_memory_history(
                            &memory_id, Some(&text), None, "DELETE",
                        ));
                    }
                    Err(e) => warn!("smart_add: failed to delete memory {memory_id}: {e}"),
                }
            }

            MemoryAction::None => {}
        }
    }

    Ok(SmartAddResult { added, updated, deleted, facts_extracted: facts.len() })
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

async fn embed_opt(
    driver: Option<&(dyn EmbeddingDriver + Send + Sync)>,
    text: &str,
) -> Option<Vec<f32>> {
    driver?.embed_one(text).await.ok()
}

fn parse_memory_id(s: &str) -> Option<MemoryId> {
    match uuid::Uuid::parse_str(s) {
        Ok(id) => Some(MemoryId(id)),
        Err(_) => {
            warn!("smart_add: invalid UUID '{s}' — skipping");
            None
        }
    }
}

fn log_history_error(result: OpenFangResult<()>) {
    if let Err(e) = result {
        warn!("smart_add: failed to write memory history: {e}");
    }
}