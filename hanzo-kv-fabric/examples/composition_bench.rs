// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! Non-prefix RAG KV composition demo at the fabric level
//! (HIP-1345 §2.7, Bounty #19 / engine#90, CacheBlend).
//!
//! Simulates a two-turn RAG session where the retrieved documents SHUFFLE
//! between turns (docB before docA on turn 1, plus one fresh docC). Strict
//! prefix caching only matches the system prompt and must re-prefill every
//! document; the composition planner reuses docA/docB wherever they land,
//! recomputing only a small window at each reused chunk start.
//!
//! Run: `cargo run --release -p hanzo-kv-fabric --example composition_bench`

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use hanzo_kv_fabric::composition::{
    ChunkMeta, CompositionDecision, CompositionPlanner, CompositionRequest, RecomputeReason,
};
use hanzo_kv_fabric::compute_block_digest;

const TOKENS_PER_DOC: u64 = 256;
const SYSTEM_TOKENS: u64 = 64;
const QUESTION_TOKENS: u64 = 64;
const RECOMPUTE_WINDOW: u32 = 16;

fn tag_hash(tag: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    hasher.finish()
}

fn tag_sha(tag: &str, offset: u64, tokens: u64) -> [u8; 32] {
    let mut buf = Vec::new();
    buf.extend_from_slice(tag.as_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
    buf.extend_from_slice(&tokens.to_le_bytes());
    compute_block_digest(&buf)
}

fn doc_chunk(tag: &str, published_offset: u64) -> ChunkMeta {
    ChunkMeta {
        chunk_hash: tag_hash(tag),
        published_offset,
        num_tokens: TOKENS_PER_DOC,
        sha256: tag_sha(tag, published_offset, TOKENS_PER_DOC),
    }
}

fn system_chunk(published_offset: u64) -> ChunkMeta {
    ChunkMeta {
        chunk_hash: tag_hash("system"),
        published_offset,
        num_tokens: SYSTEM_TOKENS,
        sha256: tag_sha("system", published_offset, SYSTEM_TOKENS),
    }
}

fn main() {
    let planner = CompositionPlanner::new(RECOMPUTE_WINDOW);

    // Turn 0: system + docA + docB + question — everything prefilled, chunks cached.
    let turn0_total = SYSTEM_TOKENS + 2 * TOKENS_PER_DOC + QUESTION_TOKENS;

    // Turn 1: system + docB + docA + docC + question — retrieval order shuffled,
    // one fresh document.
    let turn1_total = SYSTEM_TOKENS + 3 * TOKENS_PER_DOC + QUESTION_TOKENS;

    // Published on turn 0:
    let published = vec![
        system_chunk(0),
        doc_chunk("docA", SYSTEM_TOKENS),
        doc_chunk("docB", SYSTEM_TOKENS + TOKENS_PER_DOC),
    ];

    // Turn 1 layout: system@0, docB@64, docA@320, docC@576, question@832.
    let requests = vec![
        CompositionRequest {
            chunk_hash: tag_hash("system"),
            current_offset: 0,
            recompute_prefix_tokens: Some(0), // identical prefix, nothing to restore
        },
        CompositionRequest {
            chunk_hash: tag_hash("docB"),
            current_offset: SYSTEM_TOKENS,
            recompute_prefix_tokens: None,
        },
        CompositionRequest {
            chunk_hash: tag_hash("docA"),
            current_offset: SYSTEM_TOKENS + TOKENS_PER_DOC,
            recompute_prefix_tokens: None,
        },
        CompositionRequest {
            chunk_hash: tag_hash("docC"), // never cached
            current_offset: SYSTEM_TOKENS + 2 * TOKENS_PER_DOC,
            recompute_prefix_tokens: None,
        },
    ];

    println!("CacheBlend composition demo: 2 turns, 256-token docs, shuffled retrieval order");
    println!("=====================================================================");
    println!("turn 0 prompt: {turn0_total} tokens (system + docA + docB + question)");
    println!("turn 1 prompt: {turn1_total} tokens (system + docB + docA + docC + question)");

    // Strict prefix caching: turn 1 shares only the system prefix with
    // turn 0, so everything after it is re-prefilled.
    let prefix_prefill = turn1_total - SYSTEM_TOKENS;
    println!("strict prefix cache           : {SYSTEM_TOKENS}/{turn1_total} reusable, prefill {prefix_prefill} tokens");

    let plan = planner.plan(&published, &requests);
    let mut reused = Vec::new();
    let mut recomputed = Vec::new();
    for d in &plan.decisions {
        match d {
            CompositionDecision::Reuse {
                chunk_hash,
                position_shift,
                recompute_window,
                ..
            } => reused.push((*chunk_hash, *position_shift, *recompute_window)),
            CompositionDecision::Recompute { chunk_hash, reason } => {
                recomputed.push((*chunk_hash, reason.clone()))
            }
        }
    }

    let window_tokens: u64 = reused.iter().map(|(_, _, w)| u64::from(*w)).sum();
    let uncached_doc_tokens = TOKENS_PER_DOC; // docC has no cached K/V
    let fresh_question = QUESTION_TOKENS; // uncovered range: caller prefills
    let composition_compute = window_tokens + uncached_doc_tokens + fresh_question;

    println!(
        "composition planner           : {} reuse, {} recompute, order_violations {}",
        reused.len(),
        recomputed.len(),
        plan.order_violations
    );
    for (hash, shift, window) in &reused {
        let tag = ["system", "docA", "docB", "docC"]
            .iter()
            .find(|t| tag_hash(t) == *hash)
            .copied()
            .unwrap_or("?");
        println!("  reuse {tag:<7} shift {shift:+5} window {window} tokens");
    }
    for (hash, reason) in &recomputed {
        let tag = ["system", "docA", "docB", "docC"]
            .iter()
            .find(|t| tag_hash(t) == *hash)
            .copied()
            .unwrap_or("?");
        let why = match reason {
            RecomputeReason::UnknownChunk => "never cached",
            RecomputeReason::EmptyChunk => "empty chunk",
            RecomputeReason::AmbiguousDuplicate => "ambiguous metadata",
            RecomputeReason::OffsetOverflow => "offset overflow",
        };
        println!("  recompute {tag:<5} ({why})");
    }
    println!(
        "reusable from cache           : {} tokens (planner) vs {SYSTEM_TOKENS} (prefix-only)",
        plan.reusable_tokens
    );
    println!(
        "total compute (composition)   : {composition_compute} tokens = {window_tokens} windows + {uncached_doc_tokens} uncached doc + {fresh_question} fresh question"
    );
    println!(
        "compute reduction vs prefix   : {:.1}%",
        (1.0 - composition_compute as f64 / prefix_prefill as f64) * 100.0
    );
    println!();
    println!("Shuffled documents hit a strict prefix cache at exactly 0 tokens past");
    println!("the system prompt; chunk composition reuses them wherever they land");
    println!("and pays only a 16-token head window per reused doc (HIP-1345 §2.7).");
}
