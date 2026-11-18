// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! KV-residency-aware request scheduling for the KV fabric (HIP-1345 §2.3,
//! Bounty #13 / engine#94, Bidaw <https://www.usenix.org/conference/fast26/presentation/hu-shipeng>).
//!
//! Multi-turn conversations keep historical KVs in a two-tier storage system
//! (host memory + SSD). Naive arrival-order (FIFO) serving lets a cold SSD
//! load sit at the head of the queue and block every warm request behind it.
//! Bidaw's first mechanism — the compute→storage direction — schedules
//! requests with KV-loading latency awareness: separate requests whose KVs
//! reside in different storage layers, and reorder within each layer by KV
//! size so the engine stays busy while the slow tier drains.
//!
//! Metric note (important for honest reading): on a strictly sequential
//! server the *last* request always waits for the whole makespan regardless
//! of order, so max/p99 TTFT are order-invariant. What reordering buys is
//! **mean TTFT** and — the interactive-serving story — fast tiers never
//! waiting behind cold loads, reported as per-tier worst TTFT. The
//! simulation reports mean, p50, p95, max, per-tier worsts, and goodput.
//!
//! This module is the fabric-level scheduling *policy* layer:
//!
//! - [`ResidencyTier`]: where a session's historical KVs live right now
//!   (resident on device, host memory, or cold on SSD).
//! - [`SchedRequest`]: one multi-turn request with its KV footprint and tier.
//! - [`ResidencyScheduler`]: deterministic ordering (tier buckets, size-rank
//!   within bucket) plus a bounded sequential-server simulation.
//!
//! What is deliberately NOT here (see the PR's HIP-1345 update): Bidaw's
//! second mechanism (response-content access-pattern prediction for
//! eviction) and mechanism three (selective storage-efficient caching) need
//! response content and a storage trace, which a fabric scheduling policy
//! does not see. Those are documented as out-of-scope follow-ups.

/// Where a session's historical KVs reside when the request is scheduled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResidencyTier {
    /// KVs fully resident on device (no load I/O; ideal tier).
    Resident,
    /// KVs in host memory (fast load path).
    Host,
    /// KVs cold on SSD (slow load path; the head-of-line hazard).
    Ssd,
}

impl ResidencyTier {
    /// Rank used to order tiers: resident first (zero load cost), then
    /// host, then SSD — the slow tier drains last so its latency does not
    /// block the fast tiers.
    pub fn rank(self) -> u8 {
        match self {
            ResidencyTier::Resident => 0,
            ResidencyTier::Host => 1,
            ResidencyTier::Ssd => 2,
        }
    }

    /// Estimated load latency in nanoseconds for `kv_bytes` at this tier's
    /// effective bandwidth. Resident KVs never load.
    pub fn load_ns(self, kv_bytes: u64, cfg: &SchedulerConfig) -> u64 {
        match self {
            ResidencyTier::Resident => 0,
            ResidencyTier::Host => kv_bytes.saturating_mul(1_000_000_000) / cfg.host_load_bps.max(1),
            ResidencyTier::Ssd => kv_bytes.saturating_mul(1_000_000_000) / cfg.ssd_load_bps.max(1),
        }
    }
}

/// One multi-turn conversational request entering the serving queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedRequest {
    /// Stable session identity (multi-turn history owner).
    pub session: u64,
    /// Total KV bytes the engine must have resident to prefill this turn.
    pub kv_bytes: u64,
    /// Where those KVs live right now.
    pub tier: ResidencyTier,
    /// Arrival order (queue trace index), preserved for fairness reporting.
    pub arrival: usize,
}

/// Scheduler knobs: per-tier load bandwidth (bytes/sec) and prefill compute
/// rate (nanoseconds per KV byte — the engine-side token-mixing cost).
///
/// Defaults land in the Bidaw regime where load latency is a visible
/// fraction of cold-request cost: prefill ≈ 1 ns/byte (2 GiB → ~2.1 s) vs
/// SSD load at 2 GB/s (2 GiB → ~1.07 s) and host load at 20 GB/s
/// (2 GiB → ~0.11 s). Engine integration must replace these with measured
/// Hanzo-hardware rates.
#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// Host-memory load bandwidth, bytes per second.
    pub host_load_bps: u64,
    /// SSD load bandwidth, bytes per second.
    pub ssd_load_bps: u64,
    /// Prefill compute rate, nanoseconds per KV byte.
    pub prefill_ns_per_byte: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            host_load_bps: 20_000_000_000,
            ssd_load_bps: 2_000_000_000,
            prefill_ns_per_byte: 1,
        }
    }
}

/// Deterministic KV-residency-aware ordering (Bidaw mechanism 1): sort by
/// `(tier rank, kv_bytes ascending, arrival)` — tier separation first, then
/// smallest-KV-first within each tier to keep the engine busy while slow
/// loads drain. Stable: equal keys preserve arrival order.
pub fn residency_order(reqs: &mut [SchedRequest]) {
    reqs.sort_by_key(|r| (r.tier.rank(), r.kv_bytes, r.arrival));
}

/// Plain arrival-order baseline (what a queue with no residency awareness
/// does). Identity transform — kept explicit for symmetry with
/// [`residency_order`] in the bench.
pub fn fifo_order(_reqs: &mut [SchedRequest]) {
    // Arrival order is already the input order.
}

/// Per-request simulation outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestOutcome {
    /// Session the request belongs to.
    pub session: u64,
    /// Tier the request's KVs resided at.
    pub tier: ResidencyTier,
    /// Nanoseconds from queue entry to prefill completion (TTFT proxy).
    pub ttft_ns: u64,
    /// Nanoseconds spent waiting in queue ahead of this request.
    pub queue_wait_ns: u64,
    /// Nanoseconds spent loading KVs at this request's tier.
    pub load_ns: u64,
    /// Nanoseconds of prefill compute.
    pub prefill_ns: u64,
}

/// Aggregate schedule result: TTFT percentiles and goodput.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleResult {
    /// Outcomes in schedule order.
    pub outcomes: Vec<RequestOutcome>,
    /// Mean TTFT across all requests (ns).
    pub mean_ns: u64,
    /// 50th percentile TTFT (ns).
    pub p50_ns: u64,
    /// 95th percentile TTFT (ns).
    pub p95_ns: u64,
    /// Worst TTFT (ns) — order-invariant on a sequential server; reported
    /// for completeness (the fairness/starvation probe is per-tier worst).
    pub max_ns: u64,
    /// Total makespan (ns).
    pub makespan_ns: u64,
    /// Goodput: completed requests per second (fractional — a 24-request
    /// run at ~11 s is ~2.1 rps).
    pub goodput_rps: f64,
}

/// Simulate one schedule on a sequential server: each request in schedule
/// order occupies the engine for `load_ns + prefill_ns`; TTFT is the
/// completion time from queue entry. Deterministic, allocation-light.
pub fn simulate(cfg: &SchedulerConfig, reqs: &[SchedRequest]) -> ScheduleResult {
    let mut outcomes = Vec::with_capacity(reqs.len());
    let mut clock: u64 = 0;
    for r in reqs {
        let load_ns = r.tier.load_ns(r.kv_bytes, cfg);
        let prefill_ns = r.kv_bytes.saturating_mul(cfg.prefill_ns_per_byte);
        let queue_wait_ns = clock;
        let ttft_ns = clock + load_ns + prefill_ns;
        clock = ttft_ns;
        outcomes.push(RequestOutcome {
            session: r.session,
            tier: r.tier,
            ttft_ns,
            queue_wait_ns,
            load_ns,
            prefill_ns,
        });
    }
    let mut sorted = outcomes.clone();
    sorted.sort_by_key(|o| o.ttft_ns);
    let n = sorted.len();
    let pick = |q: usize| -> u64 {
        if n == 0 {
            return 0;
        }
        let idx = ((q * n) / 100).min(n - 1);
        sorted[idx].ttft_ns
    };
    let p50 = pick(50);
    let p95 = pick(95);
    let max = n.checked_sub(1).map(|i| sorted[i].ttft_ns).unwrap_or(0);
    let total_ttft: u128 = sorted.iter().map(|o| o.ttft_ns as u128).sum();
    let mean = if n > 0 { (total_ttft / n as u128) as u64 } else { 0 };
    let makespan = clock;
    let goodput_rps = if makespan > 0 {
        n as f64 * 1e9 / makespan as f64
    } else {
        0.0
    };
    ScheduleResult {
        outcomes,
        mean_ns: mean,
        p50_ns: p50,
        p95_ns: p95,
        max_ns: max,
        makespan_ns: makespan,
        goodput_rps,
    }
}

/// The scheduler facade: order then simulate in one call.
pub struct ResidencyScheduler;

impl ResidencyScheduler {
    /// Bidaw-style: residency order + simulation.
    pub fn run_residency(cfg: &SchedulerConfig, mut reqs: Vec<SchedRequest>) -> ScheduleResult {
        residency_order(&mut reqs);
        simulate(cfg, &reqs)
    }

    /// Baseline: arrival (FIFO) order + simulation.
    pub fn run_fifo(cfg: &SchedulerConfig, reqs: Vec<SchedRequest>) -> ScheduleResult {
        simulate(cfg, &reqs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reqs() -> Vec<SchedRequest> {
        vec![
            SchedRequest { session: 1, kv_bytes: 2_000_000_000, tier: ResidencyTier::Ssd, arrival: 0 },
            SchedRequest { session: 2, kv_bytes: 10_000_000, tier: ResidencyTier::Resident, arrival: 1 },
            SchedRequest { session: 3, kv_bytes: 50_000_000, tier: ResidencyTier::Host, arrival: 2 },
            SchedRequest { session: 4, kv_bytes: 1_000_000, tier: ResidencyTier::Resident, arrival: 3 },
        ]
    }

    #[test]
    fn order_is_tier_then_size() {
        let mut v = reqs();
        residency_order(&mut v);
        assert_eq!(v[0].session, 4); // resident, smallest
        assert_eq!(v[1].session, 2); // resident
        assert_eq!(v[2].session, 3); // host
        assert_eq!(v[3].session, 1); // ssd last
    }

    #[test]
    fn resident_tier_never_loads() {
        let cfg = SchedulerConfig::default();
        assert_eq!(ResidencyTier::Resident.load_ns(1 << 40, &cfg), 0);
        assert!(ResidencyTier::Host.load_ns(1_000_000, &cfg) > 0);
        assert!(ResidencyTier::Ssd.load_ns(1_000_000, &cfg) > ResidencyTier::Host.load_ns(1_000_000, &cfg));
    }

    #[test]
    fn simulation_is_deterministic_and_sums() {
        let cfg = SchedulerConfig::default();
        let a = ResidencyScheduler::run_fifo(&cfg, reqs());
        let b = ResidencyScheduler::run_fifo(&cfg, reqs());
        assert_eq!(a, b);
        let first = &a.outcomes[0];
        assert_eq!(first.ttft_ns, first.queue_wait_ns + first.load_ns + first.prefill_ns);
        assert!(a.goodput_rps > 0.0, "goodput must be positive, got {}", a.goodput_rps);
    }

    #[test]
    fn residency_beats_fifo_on_cold_head_of_line() {
        // A huge cold SSD request arriving first must not drag the mean
        // (and warm requests' TTFT) behind it.
        let cfg = SchedulerConfig::default();
        let fifo = ResidencyScheduler::run_fifo(&cfg, reqs());
        let res = ResidencyScheduler::run_residency(&cfg, reqs());
        assert!(
            res.mean_ns < fifo.mean_ns,
            "residency mean {} must beat FIFO mean {}",
            res.mean_ns,
            fifo.mean_ns
        );
        // The warm (resident + host) requests specifically must not wait
        // behind the cold load in the residency schedule.
        let warm_wait: u64 = res
            .outcomes
            .iter()
            .filter(|o| o.tier != ResidencyTier::Ssd)
            .map(|o| o.queue_wait_ns)
            .sum();
        let warm_wait_fifo: u64 = fifo
            .outcomes
            .iter()
            .filter(|o| o.tier != ResidencyTier::Ssd)
            .map(|o| o.queue_wait_ns)
            .sum();
        assert!(
            warm_wait < warm_wait_fifo,
            "residency warm wait {} must beat FIFO warm wait {}",
            warm_wait,
            warm_wait_fifo
        );
    }

    #[test]
    fn empty_schedule_is_zeroed() {
        let cfg = SchedulerConfig::default();
        let r = ResidencyScheduler::run_fifo(&cfg, vec![]);
        assert_eq!(r.p95_ns, 0);
        assert_eq!(r.mean_ns, 0);
        assert_eq!(r.goodput_rps, 0.0);
        assert_eq!(r.makespan_ns, 0);
    }
}
