#!/usr/bin/env python3
"""A self-contained demonstration of MAML-Select on a synthetic client pool.

No dataset download and no GPU are needed.  The pool reproduces the three device
tiers of the paper, and each client's local loss reduction shrinks as it is
selected more often, which is the non-stationary utility the method is built for.

The run prints, per phase of training, the tier mix of the selections, the
participation coverage, Jain's index, and the two diagnostics the manuscript
checks: the inner-step descent of Lemma 1 and the drift increment of V_T.

    python3 examples/run_demo.py --rounds 200 --clients 100 --cohort 10
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maml_select import ClientState, MAMLSelect  # noqa: E402

# Tier 1 slow, Tier 2 medium, Tier 3 fast, matching the 20/50/30 split of the paper.
TIER_SHARE = (0.20, 0.50, 0.30)
TIER_SPEED = (1.0, 2.0, 4.0)


def build_pool(n: int, rng: np.random.Generator) -> list[ClientState]:
    tiers = np.concatenate([
        np.full(int(round(n * TIER_SHARE[0])), 0),
        np.full(int(round(n * TIER_SHARE[1])), 1),
        np.full(n - int(round(n * TIER_SHARE[0])) - int(round(n * TIER_SHARE[1])), 2),
    ])
    rng.shuffle(tiers)
    return [
        ClientState(
            client_id=i,
            loss=float(rng.uniform(1.0, 2.5)),
            grad_norm=float(rng.uniform(0.5, 3.0)),
            latency=float(rng.uniform(40.0, 80.0) / TIER_SPEED[int(tiers[i])]),
            battery_ratio=1.0,
            tier=int(tiers[i]),
        )
        for i in range(n)
    ]


def jain(counts: list[int]) -> float:
    s1, s2 = sum(counts), sum(c * c for c in counts)
    return (s1 * s1) / (len(counts) * s2) if s2 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--cohort", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--lambda-latency", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pool = build_pool(args.clients, rng)
    selector = MAMLSelect(
        num_clients=args.clients,
        cohort_size=args.cohort,
        lambda_latency=args.lambda_latency,
        seed=args.seed,
    )

    counts = [0] * args.clients
    tier_counts = [0, 0, 0]
    descent_ok = descent_seen = 0
    drift = []
    seen_before_coldstart_ends: set[int] = set()

    print(f"clients={args.clients}  cohort={args.cohort}  rounds={args.rounds}  "
          f"lambda={args.lambda_latency}  cold start={selector.cold_start_rounds} rounds\n")
    header = f"{'round':>6}{'phase':>14}{'cov%':>7}{'Jain':>8}{'T1':>6}{'T2':>6}{'T3':>6}"
    print(header + "\n" + "-" * len(header))

    for r in range(args.rounds):
        chosen, diag = selector.select(pool)

        if diag is not None:
            if not math.isnan(diag.inner_descent):
                descent_seen += 1
                descent_ok += int(diag.inner_descent <= 1e-9)
            if not math.isnan(diag.drift_increment):
                drift.append(diag.drift_increment)

        # Utility decays with use, so a client that keeps being picked stops paying off.
        reductions = []
        for cid in chosen:
            client = pool[cid]
            reduction = max(0.01, client.loss * 0.25 * math.exp(-0.15 * client.participation_count))
            client.loss = max(0.05, client.loss - reduction)
            reductions.append(reduction)

        selector.observe([pool[c].latency for c in chosen], reductions)

        for cid in chosen:
            counts[cid] += 1
            tier_counts[pool[cid].tier] += 1
        for s in pool:
            s.staleness = 0 if s.client_id in chosen else s.staleness + 1
            s.participation_count += int(s.client_id in chosen)
        if r + 1 == selector.cold_start_rounds:
            seen_before_coldstart_ends = {i for i, c in enumerate(counts) if c > 0}

        if (r + 1) % max(1, args.rounds // 10) == 0 or r == 0:
            total = max(1, sum(tier_counts))
            cov = 100.0 * sum(1 for c in counts if c > 0) / args.clients
            mode = "cold start" if selector.last_selection_mode.startswith("coverage") else "policy"
            print(f"{r+1:>6}{mode:>14}{cov:>7.0f}{jain(counts):>8.3f}"
                  f"{tier_counts[0]/total:>6.2f}{tier_counts[1]/total:>6.2f}{tier_counts[2]/total:>6.2f}")

    coverage_all = 100.0 * sum(1 for c in counts if c > 0) / args.clients

    print(f"\nfinal coverage            {coverage_all:.0f}%")
    print(f"final Jain index          {jain(counts):.3f}")
    print(f"clients seen by cold start {len(seen_before_coldstart_ends)}/{args.clients}")
    if descent_seen:
        print(f"Lemma 1, inner descent    {descent_ok}/{descent_seen} rounds non-increasing")
    if drift:
        print(f"drift increments          mean {np.mean(drift):+.4f}, "
              f"sum V_T {np.sum(drift):+.2f} over {len(drift)} rounds")
        print("  a non-zero V_T is what Corollary 1 carries and a fixed-objective bound ignores")


if __name__ == "__main__":
    main()
