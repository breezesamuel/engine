// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! Non-prefix RAG KV composition planning for the KV fabric (HIP-1345 §2.7,
//! Bounty #19 / engine#98, CacheBlend <https://arxiv.org/abs/2405.16444>).
//!
//! Hanzo's prefix cache reuses K/V only when a *common prefix* matches. RAG
//! workloads break that model: retrieved documents land at different prompt
//! positions per query, so the same cached document bytes miss entirely.
//! CacheBlend's answer is chunk-level composition — reuse each retrieved
//! document's K/V wherever it lands, and selectively recompute only a small
//! window at the start of each reused chunk so cross-chunk attention
//! dependencies are restored.
//!
//! This module is the fabric-level *planning* layer for that strategy:
//!
//! - [`ChunkMeta`]: planner input describing one reusable chunk (identity
//!   hash, publish-time offset, token count, integrity digest). Deliberately
//!   decoupled from any directory: the scheduler can feed entries from a
//!   `SegmentDirectory`-style lookup, a document store, or tests.
//! - [`CompositionRequest`]: one placement — reuse `chunk_hash` at
//!   `current_offset`, with the selective-recompute window size.
//! - [`CompositionPlanner`]: turns requests into a [`CompositionPlan`] of
//!   reuse/recompute decisions with accounting totals.
//!
//! Invariants enforced here (fail closed):
//!
//! - **Unknown identity ⇒ recompute**: a request whose `chunk_hash` is not
//!   among the supplied chunks, a zero-token chunk, or an ambiguous
//!   duplicate (same hash with conflicting metadata) yields
//!   [`CompositionDecision::Recompute`], never a guess.
//! - **Overflow ⇒ recompute**: position shifts that cannot be represented
//!   as `i64` are rejected as recompute rather than forced through
//!   (mirrors [`crate`]'s offset-translation fail-closed contract).
//! - **Windows are clamped**: the selective-recompute window never exceeds
//!   the chunk's token count; a window of `num_tokens` is a full recompute
//!   of that chunk.
//! - **Uncovered ranges belong to the caller**: the plan covers only the
//!   requested chunks. Any prompt range not covered by a request MUST be
//!   fully prefilled by the scheduler — the planner never assumes content
//!   it was not told about.
//!
//! What this module does NOT do: it does not move K/V, build attention
//! masks, or translate RoPE — those belong to the engine attention path
//! once the scheduler consumes the plan.

/// Metadata for one reusable chunk supplied to [`CompositionPlanner`].
///
/// `chunk_hash` is expected to be the caller's salted content hash (tenant
/// salting happens upstream of the planner; the planner treats hashes as
/// opaque identities, so two tenants never see each other's chunks).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkMeta {
    pub chunk_hash: u64,
    /// Absolute token offset of the chunk start in the cached source
    /// sequence.
    pub published_offset: u64,
    /// Total tokens the chunk covers.
    pub num_tokens: u64,
    pub sha256: [u8; 32],
}

/// Request to place one chunk's cached K/V at a new absolute offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionRequest {
    pub chunk_hash: u64,
    /// Absolute token offset where the chunk starts in the new sequence.
    pub current_offset: u64,
    /// Tokens at the start of the reused chunk that must be recomputed to
    /// restore cross-attention dependencies (CacheBlend's selective
    /// recompute). `None` uses the planner default.
    pub recompute_prefix_tokens: Option<u32>,
}

/// The planner's decision for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionDecision {
    /// The chunk is reusable at the requested position.
    Reuse {
        chunk_hash: u64,
        /// Signed shift to apply to the chunk's position-dependent state
        /// (`current_offset - published_offset`).
        position_shift: i64,
        /// Tokens recomputed at the chunk start (selective recompute).
        recompute_window: u32,
        sha256: [u8; 32],
    },
    /// The request fails closed: the covered range must be fully prefilled.
    Recompute {
        chunk_hash: u64,
        reason: RecomputeReason,
    },
}

/// Why a request was rejected (all reasons fail closed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecomputeReason {
    /// The hash was not among the supplied chunks.
    UnknownChunk,
    /// The chunk declares zero tokens.
    EmptyChunk,
    /// The same hash appears with conflicting metadata.
    AmbiguousDuplicate,
    /// The position shift is not representable as `i64`.
    OffsetOverflow,
}

/// Result of planning a batch of requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionPlan {
    /// One decision per request, in request order.
    pub decisions: Vec<CompositionDecision>,
    /// Tokens served from cache (excludes recompute windows).
    pub reusable_tokens: u64,
    /// Tokens that must be recomputed for the covered chunks (windows plus
    /// fully rejected requests).
    pub recompute_tokens: u64,
    /// Count of adjacent request pairs whose `current_offset` decreases.
    /// The scheduler MUST treat a non-zero count as a fail-closed signal:
    /// attention-path reuse requires reused chunks to preserve their
    /// relative prompt order.
    pub order_violations: usize,
}

/// Plans non-prefix chunk composition for RAG-style prompts.
#[derive(Debug, Clone, Copy)]
pub struct CompositionPlanner {
    default_recompute_window: u32,
}

impl CompositionPlanner {
    /// Create a planner whose default selective-recompute window is
    /// `default_recompute_window` tokens at each reused chunk start.
    pub fn new(default_recompute_window: u32) -> Self {
        Self {
            default_recompute_window,
        }
    }

    /// Plan `requests` against the known `chunks`.
    ///
    /// Requests are processed in the order given; decisions align with
    /// requests one-to-one. See the module docs for the fail-closed
    /// invariants.
    pub fn plan(&self, chunks: &[ChunkMeta], requests: &[CompositionRequest]) -> CompositionPlan {
        let mut index = std::collections::HashMap::with_capacity(chunks.len());
        let mut poisoned = std::collections::HashSet::new();
        for meta in chunks {
            match index.entry(meta.chunk_hash) {
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(meta.clone());
                }
                std::collections::hash_map::Entry::Occupied(slot) => {
                    if slot.get() != meta {
                        // Conflicting metadata for one identity: poison the
                        // hash so any request on it fails closed.
                        poisoned.insert(meta.chunk_hash);
                        slot.remove();
                    }
                }
            }
        }

        let mut decisions = Vec::with_capacity(requests.len());
        let mut reusable_tokens = 0u64;
        let mut recompute_tokens = 0u64;
        let mut order_violations = 0usize;

        for (i, request) in requests.iter().enumerate() {
            if i > 0 && request.current_offset < requests[i - 1].current_offset {
                order_violations += 1;
            }

            if poisoned.contains(&request.chunk_hash) {
                decisions.push(CompositionDecision::Recompute {
                    chunk_hash: request.chunk_hash,
                    reason: RecomputeReason::AmbiguousDuplicate,
                });
                continue;
            }

            let meta = index.get(&request.chunk_hash);
            let Some(meta) = meta else {
                // Unknown identity: the caller prefills the range it asked
                // about; the planner does not charge tokens it cannot see.
                decisions.push(CompositionDecision::Recompute {
                    chunk_hash: request.chunk_hash,
                    reason: RecomputeReason::UnknownChunk,
                });
                continue;
            };
            if meta.num_tokens == 0 {
                decisions.push(CompositionDecision::Recompute {
                    chunk_hash: request.chunk_hash,
                    reason: RecomputeReason::EmptyChunk,
                });
                continue;
            }

            let position_shift = match self.safe_shift(meta.published_offset, request.current_offset)
            {
                Some(shift) => shift,
                None => {
                    recompute_tokens += meta.num_tokens;
                    decisions.push(CompositionDecision::Recompute {
                        chunk_hash: request.chunk_hash,
                        reason: RecomputeReason::OffsetOverflow,
                    });
                    continue;
                }
            };

            let window = self.window_for(request, meta.num_tokens);
            reusable_tokens += meta.num_tokens - u64::from(window);
            recompute_tokens += u64::from(window);
            decisions.push(CompositionDecision::Reuse {
                chunk_hash: request.chunk_hash,
                position_shift,
                recompute_window: window,
                sha256: meta.sha256,
            });
        }

        CompositionPlan {
            decisions,
            reusable_tokens,
            recompute_tokens,
            order_violations,
        }
    }

    /// Signed position shift, failing closed on unrepresentable values.
    fn safe_shift(&self, published_offset: u64, current_offset: u64) -> Option<i64> {
        let published = i64::try_from(published_offset).ok()?;
        let current = i64::try_from(current_offset).ok()?;
        current.checked_sub(published)
    }

    /// Clamp the selective-recompute window to the chunk length.
    fn window_for(&self, request: &CompositionRequest, num_tokens: u64) -> u32 {
        let requested = request
            .recompute_prefix_tokens
            .unwrap_or(self.default_recompute_window);
        u32::try_from(num_tokens)
            .unwrap_or(u32::MAX)
            .min(requested)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(hash: u64, published_offset: u64, num_tokens: u64) -> ChunkMeta {
        let mut sha = [0u8; 32];
        sha[0] = hash as u8;
        ChunkMeta {
            chunk_hash: hash,
            published_offset,
            num_tokens,
            sha256: sha,
        }
    }

    fn request(hash: u64, current_offset: u64) -> CompositionRequest {
        CompositionRequest {
            chunk_hash: hash,
            current_offset,
            recompute_prefix_tokens: None,
        }
    }

    #[test]
    fn all_present_yields_boundary_only_recompute() {
        let planner = CompositionPlanner::new(4);
        let chunks = vec![chunk(1, 0, 100), chunk(2, 100, 100), chunk(3, 200, 100)];
        let requests = vec![
            request(1, 128),
            request(2, 256),
            request(3, 384),
        ];

        let plan = planner.plan(&chunks, &requests);
        assert_eq!(plan.decisions.len(), 3);
        for (i, d) in plan.decisions.iter().enumerate() {
            match d {
                CompositionDecision::Reuse {
                    position_shift,
                    recompute_window,
                    ..
                } => {
                    assert_eq!(*recompute_window, 4, "default window applies");
                    let expected_shift = requests[i].current_offset as i64 - chunks[i].published_offset as i64;
                    assert_eq!(*position_shift, expected_shift);
                }
                other => panic!("expected reuse, got {other:?}"),
            }
        }
        assert_eq!(plan.reusable_tokens, 3 * 96);
        assert_eq!(plan.recompute_tokens, 12);
        assert_eq!(plan.order_violations, 0);
    }

    #[test]
    fn position_shift_reports_the_move() {
        let planner = CompositionPlanner::new(0);
        let chunks = vec![chunk(7, 0, 64)];
        let plan = planner.plan(&chunks, &[request(7, 128)]);
        match &plan.decisions[0] {
            CompositionDecision::Reuse { position_shift, .. } => {
                assert_eq!(*position_shift, 128);
            }
            other => panic!("expected reuse, got {other:?}"),
        }
    }

    #[test]
    fn unknown_hash_fails_closed() {
        let planner = CompositionPlanner::new(4);
        let chunks = vec![chunk(1, 0, 100)];
        let plan = planner.plan(&chunks, &[request(999, 0)]);
        match &plan.decisions[0] {
            CompositionDecision::Recompute {
                reason: RecomputeReason::UnknownChunk,
                ..
            } => {}
            other => panic!("expected recompute, got {other:?}"),
        }
        assert_eq!(plan.reusable_tokens, 0);
    }

    #[test]
    fn empty_chunk_fails_closed() {
        let planner = CompositionPlanner::new(4);
        let chunks = vec![chunk(1, 0, 0)];
        let plan = planner.plan(&chunks, &[request(1, 0)]);
        match &plan.decisions[0] {
            CompositionDecision::Recompute {
                reason: RecomputeReason::EmptyChunk,
                ..
            } => {}
            other => panic!("expected recompute, got {other:?}"),
        }
    }

    #[test]
    fn ambiguous_duplicate_hash_fails_closed() {
        let planner = CompositionPlanner::new(4);
        let chunks = vec![
            chunk(5, 0, 100),
            ChunkMeta {
                num_tokens: 200,
                ..chunk(5, 0, 100)
            },
        ];
        let plan = planner.plan(&chunks, &[request(5, 0)]);
        match &plan.decisions[0] {
            CompositionDecision::Recompute {
                reason: RecomputeReason::AmbiguousDuplicate,
                ..
            } => {}
            other => panic!("expected recompute, got {other:?}"),
        }
        assert_eq!(plan.reusable_tokens, 0);
    }

    #[test]
    fn identical_duplicates_are_not_ambiguous() {
        let planner = CompositionPlanner::new(4);
        let chunks = vec![chunk(5, 0, 100), chunk(5, 0, 100)];
        let plan = planner.plan(&chunks, &[request(5, 10)]);
        assert!(matches!(plan.decisions[0], CompositionDecision::Reuse { .. }));
    }

    #[test]
    fn offset_overflow_fails_closed() {
        let planner = CompositionPlanner::new(4);
        let chunks = vec![chunk(1, u64::MAX, 100)];
        let plan = planner.plan(&chunks, &[request(1, 0)]);
        match &plan.decisions[0] {
            CompositionDecision::Recompute {
                reason: RecomputeReason::OffsetOverflow,
                ..
            } => {}
            other => panic!("expected recompute, got {other:?}"),
        }
        assert_eq!(plan.recompute_tokens, 100, "rejected chunk is fully charged");
    }

    #[test]
    fn window_clamps_to_chunk_length() {
        let planner = CompositionPlanner::new(64);
        let chunks = vec![chunk(1, 0, 16)];
        let plan = planner.plan(&chunks, &[request(1, 0)]);
        match &plan.decisions[0] {
            CompositionDecision::Reuse { recompute_window, .. } => {
                assert_eq!(*recompute_window, 16, "window never exceeds the chunk");
            }
            other => panic!("expected reuse, got {other:?}"),
        }
        assert_eq!(plan.reusable_tokens, 0, "fully-windowed chunk reuses nothing");
        assert_eq!(plan.recompute_tokens, 16);
    }

    #[test]
    fn per_request_window_override_wins() {
        let planner = CompositionPlanner::new(64);
        let chunks = vec![chunk(1, 0, 100)];
        let req = CompositionRequest {
            chunk_hash: 1,
            current_offset: 40,
            recompute_prefix_tokens: Some(2),
        };
        let plan = planner.plan(&chunks, &[req]);
        match &plan.decisions[0] {
            CompositionDecision::Reuse { recompute_window, .. } => assert_eq!(*recompute_window, 2),
            other => panic!("expected reuse, got {other:?}"),
        }
    }

    #[test]
    fn order_violations_are_counted_fail_closed() {
        let planner = CompositionPlanner::new(4);
        let chunks = vec![chunk(1, 0, 10), chunk(2, 10, 10)];
        let requests = vec![
            request(2, 500),
            request(1, 100), // decreases: violation
            request(1, 200), // equal is fine, then increases
        ];
        let plan = planner.plan(&chunks, &requests);
        assert_eq!(plan.order_violations, 1);
    }

    #[test]
    fn totals_stay_consistent_across_mixed_decisions() {
        let planner = CompositionPlanner::new(8);
        let chunks = vec![chunk(1, 0, 256), chunk(2, 256, 256), chunk(3, 512, 256)];
        let requests = vec![
            request(1, 0),    // reuse
            request(2, 300),  // reuse
            request(999, 0),  // unknown → not counted (caller prefills)
            request(3, 600),  // reuse
        ];
        let plan = planner.plan(&chunks, &requests);
        let reuse_count = plan
            .decisions
            .iter()
            .filter(|d| matches!(d, CompositionDecision::Reuse { .. }))
            .count();
        assert_eq!(reuse_count, 3);
        assert_eq!(plan.reusable_tokens, 3 * (256 - 8));
        assert_eq!(plan.recompute_tokens, 3 * 8);
    }
}
