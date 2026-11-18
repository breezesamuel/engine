// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: MIT

//! KV-residency-aware scheduling demo at the fabric level
//! (HIP-1345 §2.3, Bounty #13 / engine#94, Bidaw).
//!
//! Workload: 24 multi-turn conversational requests with a deterministic
//! warm/cold mix — 8 resident (KVs on device), 8 host-memory-warm, 8
//! SSD-cold, KV footprints spanning 16 MiB to 2 GiB, interleaved arrival.
//! Two schedules on identical input:
//!   A) FIFO arrival order (no residency awareness) — the baseline.
//!   B) Bidaw ordering: tier separation (resident → host → SSD), smallest-
//!      KV-first within each tier.
//! A sequential-server simulation reports TTFT mean/percentiles, per-tier
//! worsts, and goodput. Metric honesty: on a sequential server the last
//! request always waits the full makespan regardless of order, so max TTFT
//! is order-invariant — what reordering buys is mean TTFT and warm tiers
//! never waiting behind cold loads. Policy-layer only: no device I/O, no
//! attention path — those belong to the engine.
//!
//! Run: `cargo run --release --example residency_bench`

use std::time::Instant;

use hanzo_kv_residency::{ResidencyScheduler, ResidencyTier, SchedulerConfig, SchedRequest};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// Deterministic KV footprint ladder: 16 MiB .. 2 GiB.
fn kv_bytes(i: usize) -> u64 {
    let sizes: [u64; 8] = [
        16 * MIB,
        32 * MIB,
        64 * MIB,
        128 * MIB,
        256 * MIB,
        512 * MIB,
        1 * GIB,
        2 * GIB,
    ];
    sizes[i % sizes.len()]
}

/// Shuffled arrival: interleaved tiers, deterministic (no RNG).
fn arrival_order() -> Vec<SchedRequest> {
    let tiers = [ResidencyTier::Ssd, ResidencyTier::Resident, ResidencyTier::Host];
    let mut v = Vec::with_capacity(24);
    for i in 0..24 {
        v.push(SchedRequest {
            session: 1000 + i as u64,
            kv_bytes: kv_bytes(i),
            tier: tiers[i % tiers.len()],
            arrival: i,
        });
    }
    v
}

fn percentile(sorted: &mut [u64], q: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted.sort_unstable();
    sorted[((q * sorted.len()) / 100).min(sorted.len() - 1)]
}

fn report(label: &str, r: &hanzo_kv_residency::ScheduleResult, n: usize) {
    // Warm = resident + host tiers.
    let mut warm: Vec<u64> = r
        .outcomes
        .iter()
        .filter(|o| o.tier != ResidencyTier::Ssd)
        .map(|o| o.ttft_ns)
        .collect();
    let warm_p95 = percentile(&mut warm, 95) as f64 / 1e6;
    let ssd_worst: Vec<u64> = r
        .outcomes
        .iter()
        .filter(|o| o.tier == ResidencyTier::Ssd)
        .map(|o| o.ttft_ns)
        .collect();
    let ssd_max = ssd_worst.iter().max().copied().unwrap_or(0) as f64 / 1e6;

    println!("{label}:");
    println!(
        "  mean {: >10.1} ms   p50 {: >10.1} ms   p95 {: >10.1} ms   max {: >10.1} ms (order-invariant)",
        r.mean_ns as f64 / 1e6,
        r.p50_ns as f64 / 1e6,
        r.p95_ns as f64 / 1e6,
        r.max_ns as f64 / 1e6,
    );
    println!(
        "  warm-tier p95 {: >10.1} ms   ssd-tier worst {: >10.1} ms   goodput {:.2} rps",
        warm_p95,
        ssd_max,
        r.goodput_rps,
    );
    let _ = n;
}

fn main() {
    let fast = SchedulerConfig::default(); // 20 GB/s host, 2 GB/s SSD
    let slow = SchedulerConfig { ssd_load_bps: 500_000_000, ..SchedulerConfig::default() }; // older SSD
    for (label, cfg) in [("scenario 1: 2 GB/s SSD", fast), ("scenario 2: 500 MB/s SSD", slow)] {
        println!();
        println!("### {label}");
        let input = arrival_order();
        println!(
            "KV-residency scheduling demo: {} requests (8 resident / 8 host / 8 ssd), KV 16 MiB..2 GiB",
            input.len()
        );
        println!("=====================================================================");

        // FIFO baseline.
        let fifo = ResidencyScheduler::run_fifo(&cfg, input.clone());
        report("A) FIFO arrival order", &fifo, input.len());

        // Bidaw-style residency ordering.
        let start = Instant::now();
        let res = ResidencyScheduler::run_residency(&cfg, input.clone());
        let order_cost = start.elapsed();
        report("B) residency order (tier sep + size rank)", &res, input.len());
        println!(
            "  ordering cost: {} ns for {} requests ({} ns/req)",
            order_cost.as_nanos(),
            input.len(),
            order_cost.as_nanos() / input.len() as u128,
        );

        let mean_gain = fifo.mean_ns as f64 / res.mean_ns.max(1) as f64;
        let warm_fifo: Vec<u64> = fifo
            .outcomes
            .iter()
            .filter(|o| o.tier != ResidencyTier::Ssd)
            .map(|o| o.ttft_ns)
            .collect();
        let warm_res: Vec<u64> = res
            .outcomes
            .iter()
            .filter(|o| o.tier != ResidencyTier::Ssd)
            .map(|o| o.ttft_ns)
            .collect();
        let mut warm_fifo_sorted = warm_fifo;
        let mut warm_res_sorted = warm_res;
        let warm_p95_fifo = percentile(&mut warm_fifo_sorted, 95) as f64 / 1e6;
        let warm_p95_res = percentile(&mut warm_res_sorted, 95) as f64 / 1e6;
        println!();
        println!("mean TTFT improvement   : {:.2}x", mean_gain);
        println!(
            "warm-tier p95 TTFT      : {:.1} ms -> {:.1} ms ({:.2}x)",
            warm_p95_fifo,
            warm_p95_res,
            warm_p95_fifo / warm_p95_res.max(0.001),
        );
        println!(
            "goodput                 : {:.2} -> {:.2} rps (total load bytes identical by construction)",
            fifo.goodput_rps,
            res.goodput_rps,
        );
    }
    println!();
    println!("FIFO lets cold SSD loads sit ahead of warm requests in the queue;");
    println!("tier separation + smallest-KV-first within tier keeps the engine on");
    println!("fast tiers while the slow tier drains — Bidaw mechanism 1 (compute");
    println!("-> storage awareness) at the fabric scheduling-policy layer (HIP");
    println!("1345 §2.3). Mechanism 2 (response-predicted eviction) is out of");
    println!("scope. max TTFT is order-invariant on a sequential server and is");
    println!("reported as such — the win is mean and warm-tier latency.");
}
