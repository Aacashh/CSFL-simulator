"""Participation fairness with and without the forced cold-start rounds.

MAML-Select opens every run with ceil(N/K) rounds of deterministic round-robin
coverage. Those rounds alone guarantee that every client is selected once, so a
coverage number computed over the whole run partly measures the warm-up rather
than the learned policy. This script recomputes coverage and the Jain index over
the post-warm-up rounds only, which is the number the fairness claim should rest
on.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import load


def jain(counts):
    counts = list(counts)
    total = sum(counts)
    sq = sum(c * c for c in counts)
    if sq == 0:
        return 0.0
    return (total * total) / (len(counts) * sq)


def participation(run, skip_rounds=0, horizon=None):
    """Per-client selection counts over rounds in (skip_rounds, horizon]."""
    n = int(run["config"].get("total_clients") or len(run["participation_counts"]) or 100)
    counts = [0] * n
    rounds = 0
    for e in run["metrics"]:
        r = e.get("round")
        sel = e.get("selected_clients")
        if r is None or r < 0 or sel is None:
            continue
        if horizon is not None and r > horizon:
            continue
        if r < skip_rounds:
            continue
        rounds += 1
        for c in sel:
            if 0 <= int(c) < n:
                counts[int(c)] += 1
    return counts, rounds


def warmup_rounds(run):
    n = int(run["config"].get("total_clients") or 100)
    k = int(run["config"].get("clients_per_round") or 10)
    return int(math.ceil(n / max(1, k)))


def summarise(runs, horizon=None):
    out = []
    for r in runs:
        w = warmup_rounds(r)
        all_c, all_r = participation(r, 0, horizon)
        post_c, post_r = participation(r, w, horizon)
        out.append({
            "label": r["label"],
            "method": r["method"],
            "seed": r["seed"],
            "warmup": w,
            "rounds_all": all_r,
            "rounds_post": post_r,
            "cov_all": 100.0 * sum(1 for c in all_c if c > 0) / len(all_c),
            "cov_post": 100.0 * sum(1 for c in post_c if c > 0) / len(post_c),
            "jain_all": jain(all_c),
            "jain_post": jain(post_c),
        })
    return out


if __name__ == "__main__":
    runs = load.load_all()
    cif = [r for r in runs
           if r["experiment"] == "cifar100_benchmarks" and r["method"] == "MAML-Select"]
    conv = [r for r in runs if r["experiment"].startswith("conv_")]
    fash = [r for r in runs
            if r["experiment"] == "feature_ablation" and r["method"] == "full"]

    print("=== MAML-Select, CIFAR-100 benchmark runs, horizon 150")
    print(f"{'seed':>6}{'warmup':>8}{'rounds':>8}{'post':>7}"
          f"{'cov all':>10}{'cov post':>10}{'jain all':>10}{'jain post':>11}")
    for s in summarise(cif, horizon=150):
        print(f"{s['seed']:>6}{s['warmup']:>8}{s['rounds_all']:>8}{s['rounds_post']:>7}"
              f"{s['cov_all']:>10.1f}{s['cov_post']:>10.1f}"
              f"{s['jain_all']:>10.3f}{s['jain_post']:>11.3f}")
    for key in ("cov_all", "cov_post", "jain_all", "jain_post"):
        m, sd, n = load.mean_sd([s[key] for s in summarise(cif, horizon=150)])
        print(f"    {key:<10} mean {m:.3f} sd {sd:.3f} over n={n}")

    print("\n=== convergence runs, full horizon")
    print(f"{'experiment':<16}{'seed':>6}{'rounds':>8}{'post':>7}"
          f"{'cov all':>10}{'cov post':>10}{'jain all':>10}{'jain post':>11}")
    for r, s in zip(conv, summarise(conv)):
        print(f"{r['experiment']:<16}{s['seed']:>6}{s['rounds_all']:>8}{s['rounds_post']:>7}"
              f"{s['cov_all']:>10.1f}{s['cov_post']:>10.1f}"
              f"{s['jain_all']:>10.3f}{s['jain_post']:>11.3f}")

    print("\n=== Fashion-MNIST, full state vector runs from the feature ablation")
    print(f"{'seed':>6}{'rounds':>8}{'post':>7}"
          f"{'cov all':>10}{'cov post':>10}{'jain all':>10}{'jain post':>11}")
    for s in summarise(fash):
        print(f"{s['seed']:>6}{s['rounds_all']:>8}{s['rounds_post']:>7}"
              f"{s['cov_all']:>10.1f}{s['cov_post']:>10.1f}"
              f"{s['jain_all']:>10.3f}{s['jain_post']:>11.3f}")
    for key in ("cov_post", "jain_all", "jain_post"):
        m, sd, n = load.mean_sd([s[key] for s in summarise(fash)])
        print(f"    {key:<10} mean {m:.3f} sd {sd:.3f} over n={n}")
