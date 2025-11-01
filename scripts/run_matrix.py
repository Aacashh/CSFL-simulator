from __future__ import annotations
import argparse
from pathlib import Path
import json
from itertools import product

from csfl_simulator.core.simulator import FLSimulator, SimConfig
from csfl_simulator.core.utils import ROOT


BASELINES = [
    "heuristic.random",
    "system_aware.fedcs",
    "system_aware.tifl",
    "system_aware.oort",
    "ml.bandit.epsilon_greedy",
    "ml.bandit.linucb",
    "ml.fedcor",
]

PROPOSED = [
    "ml.pareto_rl",
    "ml.dp_eig",
    "ml.gnn_dpp",
]


def run_once(dataset: str, partition: str, alpha: float, total_clients: int, K: int, rounds: int,
             dp_sigma: float, dp_eps_round: float, dp_clip: float, method: str, seed: int, fast: bool) -> dict:
    cfg = SimConfig(
        dataset=dataset,
        partition=("dirichlet" if partition == "dirichlet" else partition),
        dirichlet_alpha=float(alpha),
        total_clients=int(total_clients),
        clients_per_round=int(K),
        rounds=int(rounds),
        local_epochs=1,
        batch_size=32,
        lr=0.01,
        model=("CNN-MNIST" if "MNIST" in dataset.upper() else "LightCIFAR"),
        device="auto",
        seed=int(seed),
        fast_mode=bool(fast),
        time_budget=None,
        dp_sigma=float(dp_sigma),
        dp_epsilon_per_round=float(dp_eps_round),
        dp_clip_norm=float(dp_clip),
        track_grad_norm=True,
        parallel_clients=0,
    )
    sim = FLSimulator(cfg)
    res = sim.run(method_key=method)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CIFAR10")
    ap.add_argument("--partition", default="dirichlet", choices=["iid", "dirichlet", "shards"]) 
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--total_clients", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--methods", default="all", help="comma-separated or 'all' for baselines+proposed")
    args = ap.parse_args()

    methods = []
    if args.methods == "all":
        methods = BASELINES + PROPOSED
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    grid = {
        "K": [50, 100, 200],
        "dp_sigma": [0.0, 0.5],
        "dp_eps_round": [0.0, 0.1],
        "dp_clip": [0.0, 1.0],
        "alpha": [args.alpha],
        "seed": list(range(args.seeds)),
    }

    export_dir = ROOT / "artifacts" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for method, K, sigma, eps_r, clip, alpha, seed in product(
        methods, grid["K"], grid["dp_sigma"], grid["dp_eps_round"], grid["dp_clip"], grid["alpha"], grid["seed"],
    ):
        print(f"Running: {method} | K={K} | sigma={sigma} | eps/rnd={eps_r} | clip={clip} | seed={seed}")
        out = run_once(
            dataset=args.dataset,
            partition=args.partition,
            alpha=alpha,
            total_clients=args.total_clients,
            K=K,
            rounds=args.rounds,
            dp_sigma=sigma,
            dp_eps_round=eps_r,
            dp_clip=clip,
            method=method,
            seed=seed,
            fast=bool(args.fast),
        )
        results.append({
            "method": method,
            "K": K,
            "sigma": sigma,
            "eps_round": eps_r,
            "clip": clip,
            "seed": seed,
            "run_id": out.get("run_id"),
            "metrics_path": str((ROOT / "artifacts" / "runs" / out.get("run_id", "")).resolve()),
        })

    out_path = export_dir / f"matrix_{args.dataset}_{args.partition}.json"
    out_path.write_text(json.dumps({"results": results}, indent=2))
    print(f"Saved matrix summary to {out_path}")


if __name__ == "__main__":
    main()


