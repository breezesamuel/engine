# HIP-1345 Status Update — Non-Prefix RAG KV Fusion (Bounty #19, PR #105)

This is the proposed status text for HIP-1345 §2.7 (Non-Prefix RAG KV Fusion &
Incremental GPU Checkpointing, Bounties #19–#20), per the live HIP text at
`hanzoai/hips/HIPs/hip-1345-agent-aware-model-serving-and-cache-reuse.md`
(§2.7 covers Bounties #19–#20; the earlier §2.4 reference in this PR's first
review cycle was a mislabel — §2.4 is Bounties #14–#15, LoRA paging).

## Proposed §2.7 addendum (status note)

> **§2.7 Status — Non-Prefix RAG KV Fusion (Bounty #19, planning layer).** A
> CacheBlend-style composition *planner* is implemented in `hanzo-kv-fabric`
> (`composition` module): it maps retrieved document chunks to reuse/recompute
> decisions across non-prefix prompt layouts, fails closed on unknown, empty,
> ambiguous, and overflow inputs (four named rejection classes), and reports
> per-decision token shifts. On a deterministic two-turn shuffled-retrieval
> fixture (system 64 + 256-token docs; turn 1 = 896 tokens with docB/docA in
> swapped positions), the strict prefix-cache baseline reuses 64/896 tokens
> (prefill 832) while the planner reuses 544/896 (8.5×), recompute decisions 1
> (never-cached doc), silent order violations 0, total compute 352 tokens —
> a 57.7% reduction against the prefix-only baseline on identical input. Raw
> numbers: `results/cacheblend-results.json` in the PR. These are
> planning-layer token-accounting figures; the upstream CacheBlend TTFT and
> answer-parity figures remain the original authors' own and are NOT treated
> as Hanzo expectations until the attention-path integration is measured on
> Hanzo hardware.

## What this PR proves

- Non-prefix reuse restores the majority of cached tokens under shuffled
  retrieval order, where strict prefix caching collapses to the system prompt.
- The planner is fail-closed: every order violation is either detected and
  handled via a named rejection class or recomputed — zero silent stale-KV
  admissions (each rejection class has a dedicated test).
- The cross-block fusion cost is bounded and explicit (16-token head window
  per reused chunk), so the engine can budget recompute work deterministically.

## Explicit out-of-scope statement

This PR does **not** cover:

1. **KV blending math** — CacheBlend's selective token-level recomputation
   inside fused blocks is the attention-path integration (hanzo-engine), not
   this planning layer.
2. **Cross-segment attention routing** — how fused windows are stitched into
   the running attention computation is Phase 2.
3. **TTFT / answer-quality parity measurement** — requires the engine's
   attention path; the planner measures token accounting only.
4. **Multi-tenant pinning policy under budget pressure** — interplay with the
   §2.4 paging registry is future work.
5. **Runtime eviction under memory pressure** — the planner assumes cached
   blocks remain resident; eviction decisions belong to the storage tier.
6. **Bounty #20 (incremental GPU checkpointing)** — separate deliverable.

## Follow-ups for §2.7 completion

- Hanzo-engine attention-path integration (mask/window enforcement + real-KV
  stitching) to produce Hanzo-expectation TTFT and answer-parity numbers.
- Cross-layer interaction test with the §2.4 paging registry (planner decisions
  when a planned-for-reuse block has been evicted mid-turn).
