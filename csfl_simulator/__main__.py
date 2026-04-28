"""CLI entry point for CSFL Simulator.

Usage:
    python -m csfl_simulator run --method heuristic.random --dataset MNIST --rounds 10
    python -m csfl_simulator run --name exp1_baseline --method heuristic.random --dataset CIFAR-10 --rounds 20
    python -m csfl_simulator compare --name noniid_study --methods "heuristic.random,system_aware.oort" --dataset MNIST
    python -m csfl_simulator plot --run artifacts/runs/exp1_baseline_20260321-120000 --metrics accuracy,loss
    python -m csfl_simulator plot --run artifacts/runs/exp1_baseline_20260321-120000 --format eps
    python -m csfl_simulator list-methods
    python -m csfl_simulator list-runs
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from csfl_simulator.core.simulator import FLSimulator, SimConfig
from csfl_simulator.selection.registry import MethodRegistry


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_sim_args(p: argparse.ArgumentParser):
    """Add all SimConfig fields as CLI flags."""
    p.add_argument("--name", default=None,
                    help="Name for this run (creates artifacts/runs/<name>_<timestamp>/ folder)")
    p.add_argument("--dataset", default="MNIST", help="Dataset name (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100)")
    p.add_argument("--partition", default="iid", choices=["iid", "dirichlet", "label-shard"], help="Partition strategy")
    p.add_argument("--dirichlet-alpha", type=float, default=0.5, help="Dirichlet alpha (lower = more non-IID)")
    p.add_argument("--shards-per-client", type=int, default=2, help="Shards per client for label-shard partition")
    p.add_argument("--size-distribution", default="uniform", choices=["uniform", "lognormal", "power_law"],
                    help="Data quantity heterogeneity distribution")
    p.add_argument("--total-clients", type=int, default=10, help="Total number of clients")
    p.add_argument("--clients-per-round", type=int, default=3, help="Clients selected per round")
    p.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    p.add_argument("--local-epochs", type=int, default=1, help="Local training epochs per round")
    p.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--model", default="CNN-MNIST", help="Model architecture (CNN-MNIST, LightCNN, ResNet18)")
    p.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    # Smoke-test mode (formerly --fast-mode). Default OFF — only use for debugging. The old
    # --fast-mode / --no-fast-mode flag names are kept as aliases so existing shell scripts keep
    # working; internally they set cfg.smoke_test_mode.
    p.add_argument("--fast-mode", "--smoke-test-mode",
                   dest="smoke_test_mode", action="store_true", default=False,
                   help="Smoke-test mode: 1 training step + 1-2 distill batches per round (debug only)")
    p.add_argument("--no-fast-mode", "--no-smoke-test-mode",
                   dest="smoke_test_mode", action="store_false",
                   help="Disable smoke-test mode (default)")
    p.add_argument("--time-budget", type=float, default=None, help="Time budget per round")
    p.add_argument("--energy-budget", type=float, default=None, help="Energy budget per round")
    p.add_argument("--bytes-budget", type=float, default=None, help="Bytes budget per round")
    p.add_argument("--track-grad-norm", action="store_true", default=False, help="Track gradient norms")
    p.add_argument("--parallel-clients", type=int, default=-1,
                    help="Parallel client training (-1=auto [default], 0=off, N>0=fixed)")
    p.add_argument("--output", "-o", default=None, help="Output JSON path (auto-generated if not specified)")
    p.add_argument("--dp-sigma", type=float, default=0.0, help="DP Gaussian noise sigma")
    p.add_argument("--dp-epsilon-per-round", type=float, default=0.0, help="DP epsilon budget per round")
    p.add_argument("--dp-clip-norm", type=float, default=0.0, help="DP gradient clip norm")
    # --- Paradigm & Federated Distillation (FD) ---
    p.add_argument("--paradigm", default="fl", choices=["fl", "fd"],
                    help="Training paradigm: fl (Federated Learning) or fd (Federated Distillation)")
    p.add_argument("--public-dataset", default="same",
                    help="Public dataset for FD logit exchange (same=test split, STL-10, FMNIST)")
    p.add_argument("--public-dataset-size", type=int, default=2000,
                    help="Number of public samples for FD (paper: 2000)")
    p.add_argument("--distillation-epochs", type=int, default=2,
                    help="Distillation steps per round (paper: 2)")
    p.add_argument("--distillation-batch-size", type=int, default=500,
                    help="Batch size for distillation (paper: 500)")
    p.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for KL divergence")
    p.add_argument("--distillation-lr", type=float, default=0.001,
                    help="Learning rate for distillation (paper: Adam 0.001)")
    p.add_argument("--dynamic-steps", action="store_true", default=True,
                    help="Enable dynamic training steps K_r (FedTSKD)")
    p.add_argument("--no-dynamic-steps", dest="dynamic_steps", action="store_false",
                    help="Disable dynamic training steps")
    p.add_argument("--dynamic-steps-base", type=int, default=5,
                    help="Initial multiplier for dynamic steps (paper: 5)")
    p.add_argument("--dynamic-steps-period", type=int, default=25,
                    help="Rounds per step decrease (paper: 25)")
    p.add_argument("--model-heterogeneous", action="store_true", default=False,
                    help="Enable model heterogeneity (different architectures per client)")
    p.add_argument("--model-pool", default="",
                    help="Comma-separated model names for heterogeneity (e.g. FD-CNN1,FD-CNN2,FD-CNN3)")
    p.add_argument("--channel-noise", action="store_true", default=False,
                    help="Enable mMIMO channel noise simulation")
    p.add_argument("--n-bs-antennas", type=int, default=64,
                    help="Base station antennas N_BS (paper: 64)")
    p.add_argument("--n-device-antennas", type=int, default=1,
                    help="Device antennas N_D (paper: 1)")
    p.add_argument("--ul-snr-db", type=float, default=-8.0,
                    help="Uplink SNR in dB (paper: -8)")
    p.add_argument("--dl-snr-db", type=float, default=-20.0,
                    help="Downlink SNR in dB (paper: -20)")
    p.add_argument("--quantization-bits", type=int, default=8,
                    help="Logit quantization bits (paper: 8)")
    p.add_argument("--group-based", action="store_true", default=False,
                    help="Enable FedTSKD-G group-based channel-aware FD")
    p.add_argument("--channel-threshold", type=float, default=0.5,
                    help="Channel quality threshold for good/bad groups")
    p.add_argument("--fd-optimizer", default="adam", choices=["adam", "sgd"],
                    help="Optimizer for FD local training (paper: adam)")
    # Performance tuning
    p.add_argument("--eval-every", type=int, default=10,
                    help="Evaluate every N rounds (default: 10; final round always evaluated)")
    p.add_argument("--use-amp", dest="use_amp", action="store_true", default=True,
                    help="Enable mixed precision (AMP). Default: on when CUDA is detected.")
    p.add_argument("--no-amp", dest="use_amp", action="store_false",
                    help="Disable mixed precision (AMP).")
    p.add_argument("--performance-mode", dest="performance_mode", action="store_true", default=True,
                    help="Enable cuDNN autotune (torch.backends.cudnn.benchmark). Default: on. "
                         "Preserves seed-level reproducibility but not bit-for-bit.")
    p.add_argument("--no-performance-mode", dest="performance_mode", action="store_false",
                    help="Disable cuDNN autotune for bit-for-bit reproducibility (slower).")
    p.add_argument("--use-torch-compile", action="store_true", default=False,
                    help="Wrap client and server models in torch.compile (tier 3.1 — experimental).")
    p.add_argument("--channels-last", action="store_true", default=False,
                    help="Use channels_last memory format for CNN models (tier 3.2 — Ampere+).")
    p.add_argument("--eval-subsample", type=int, default=0,
                    help="Subsample size for intermediate eval rounds (tier 3.3). 0 = use full test set.")
    p.add_argument("--profile", action="store_true", default=False,
                    help="Print per-round per-phase timing breakdown to stdout (timings are always recorded in metrics.json)")
    # --- SCOPE-FD coefficient overrides ---
    # When unset (None), the registry/preset values apply (αu=0.3, αd=0.1).
    # The selector does not enforce αu + αd < 1; values that violate the
    # dominance-margin constraint are accepted as-is for ablation studies.
    p.add_argument("--scope-au", type=float, default=None,
                    help="Override SCOPE-FD αu (server-uncertainty bonus weight). Default: preset (0.3).")
    p.add_argument("--scope-ad", type=float, default=None,
                    help="Override SCOPE-FD αd (per-round diversity penalty weight). Default: preset (0.1).")


def _args_to_config(args) -> SimConfig:
    """Convert parsed CLI args to SimConfig."""
    return SimConfig(
        name=getattr(args, "name", None),
        dataset=args.dataset,
        partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha,
        shards_per_client=args.shards_per_client,
        size_distribution=args.size_distribution,
        total_clients=args.total_clients,
        clients_per_round=args.clients_per_round,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model=args.model,
        device=args.device,
        seed=args.seed,
        smoke_test_mode=args.smoke_test_mode,
        time_budget=args.time_budget,
        energy_budget=args.energy_budget,
        bytes_budget=args.bytes_budget,
        track_grad_norm=args.track_grad_norm,
        parallel_clients=args.parallel_clients,
        dp_sigma=args.dp_sigma,
        dp_epsilon_per_round=args.dp_epsilon_per_round,
        dp_clip_norm=args.dp_clip_norm,
        # Federated Distillation fields
        paradigm=getattr(args, "paradigm", "fl"),
        public_dataset=getattr(args, "public_dataset", "same"),
        public_dataset_size=getattr(args, "public_dataset_size", 2000),
        distillation_epochs=getattr(args, "distillation_epochs", 2),
        distillation_batch_size=getattr(args, "distillation_batch_size", 500),
        temperature=getattr(args, "temperature", 1.0),
        distillation_lr=getattr(args, "distillation_lr", 0.001),
        dynamic_steps=getattr(args, "dynamic_steps", True),
        dynamic_steps_base=getattr(args, "dynamic_steps_base", 5),
        dynamic_steps_period=getattr(args, "dynamic_steps_period", 25),
        model_heterogeneous=getattr(args, "model_heterogeneous", False),
        model_pool=getattr(args, "model_pool", ""),
        channel_noise=getattr(args, "channel_noise", False),
        n_bs_antennas=getattr(args, "n_bs_antennas", 64),
        n_device_antennas=getattr(args, "n_device_antennas", 1),
        ul_snr_db=getattr(args, "ul_snr_db", -8.0),
        dl_snr_db=getattr(args, "dl_snr_db", -20.0),
        quantization_bits=getattr(args, "quantization_bits", 8),
        group_based=getattr(args, "group_based", False),
        channel_threshold=getattr(args, "channel_threshold", 0.5),
        fd_optimizer=getattr(args, "fd_optimizer", "adam"),
        eval_every=getattr(args, "eval_every", 10),
        use_amp=getattr(args, "use_amp", True),
        performance_mode=getattr(args, "performance_mode", True),
        use_torch_compile=getattr(args, "use_torch_compile", False),
        channels_last=getattr(args, "channels_last", False),
        eval_subsample=getattr(args, "eval_subsample", 0),
        profile=getattr(args, "profile", False),
        scope_au=getattr(args, "scope_au", None),
        scope_ad=getattr(args, "scope_ad", None),
    )


def _progress_callback(total_rounds: int):
    """Return a progress callback that prints to stderr."""
    def cb(rnd, info):
        acc = info.get("accuracy", 0.0)
        loss = info.get("metrics", {}).get("loss", 0.0)
        print(f"  Round {rnd + 1}/{total_rounds}: acc={acc:.4f}  loss={loss:.4f}", file=sys.stderr)
    return cb


def _print_table(headers: list[str], rows: list[list], col_widths: list[int] | None = None):
    """Print a simple aligned table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    header_line = ""
    sep_line = ""
    for h, w in zip(headers, col_widths):
        header_line += h.ljust(w) + " | "
        sep_line += "-" * w + "-+-"
    print(header_line.rstrip(" | "))
    print(sep_line.rstrip("-+-"))
    for row in rows:
        line = ""
        for val, w in zip(row, col_widths):
            line += str(val).ljust(w) + " | "
        print(line.rstrip(" | "))


def _extract_summary(result: dict) -> dict:
    """Extract key summary metrics from a run result."""
    metrics = result.get("metrics", [])
    final = metrics[-1] if metrics else {}
    summary = {
        "accuracy": f"{final.get('accuracy', 0.0):.4f}",
        "loss": f"{final.get('loss', 0.0):.4f}",
        "f1": f"{final.get('f1', 0.0):.4f}",
        "fairness_gini": f"{final.get('fairness_gini', 0.0):.4f}",
        "wall_clock": f"{final.get('wall_clock', 0.0):.2f}s",
        "cum_tflops": f"{final.get('cum_tflops', 0.0):.6f}",
        "cum_comm": f"{final.get('cum_comm', 0.0):.2f}MB",
    }
    # FD-specific summary fields
    if result.get("paradigm") == "fd" or final.get("logit_comm_kb") is not None:
        summary["logit_comm_kb"] = f"{final.get('logit_comm_kb', 0.0):.2f}KB"
        summary["comm_reduction"] = f"{final.get('comm_reduction_ratio', 0.0):.4f}"
        summary["kl_divergence"] = f"{final.get('kl_divergence_avg', 0.0):.4f}"
        summary["client_acc_std"] = f"{final.get('client_accuracy_std', 0.0):.4f}"
    return summary


def _resolve_run_dir(run_path: str) -> Path:
    """Resolve a run path — accepts full path, relative path, or just a name prefix."""
    p = Path(run_path)
    if p.is_dir():
        return p
    # Try under artifacts/runs/
    from csfl_simulator.core.utils import ART_ROOT
    candidate = ART_ROOT / "runs" / run_path
    if candidate.is_dir():
        return candidate
    # Try prefix match (e.g. "exp1" matches "exp1_20260321-120000")
    runs_dir = ART_ROOT / "runs"
    if runs_dir.is_dir():
        matches = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(run_path)],
                         key=lambda d: d.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    print(f"Error: Run directory not found: {run_path}", file=sys.stderr)
    print(f"  Looked in: {runs_dir}", file=sys.stderr)
    sys.exit(1)


def _load_run_results(run_dir: Path) -> dict:
    """Load run results from a directory or standalone JSON file.

    Prefers compare_results.json (multi-method) over metrics.json (single method).
    """
    # Could be a direct JSON file
    if run_dir.suffix == ".json" and run_dir.is_file():
        with open(run_dir) as f:
            return json.load(f)
    # Standard run directory — prefer compare_results.json for multi-method runs
    compare_path = run_dir / "compare_results.json"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    if compare_path.exists():
        with open(compare_path) as f:
            data = json.load(f)
        data["run_dir"] = str(run_dir)
        return data
    if not metrics_path.exists():
        print(f"Error: No metrics.json or compare_results.json found in {run_dir}", file=sys.stderr)
        sys.exit(1)
    with open(metrics_path) as f:
        data = json.load(f)
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        data["config"] = cfg.get("config", cfg)
    data["run_dir"] = str(run_dir)
    return data


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _create_simulator(cfg: SimConfig):
    """Create the appropriate simulator based on paradigm."""
    if cfg.paradigm == "fd":
        from csfl_simulator.core.fd_simulator import FDSimulator
        return FDSimulator(cfg)
    return FLSimulator(cfg)


def cmd_run(args):
    """Run a single simulation."""
    cfg = _args_to_config(args)
    method = args.method

    paradigm_str = f" [{cfg.paradigm.upper()}]" if cfg.paradigm != "fl" else ""
    name_str = f" (name={cfg.name})" if cfg.name else ""
    print(f"Running{paradigm_str}: method={method} dataset={cfg.dataset} partition={cfg.partition} "
          f"clients={cfg.total_clients} K={cfg.clients_per_round} rounds={cfg.rounds}{name_str}", file=sys.stderr)

    sim = _create_simulator(cfg)
    t0 = time.time()
    result = sim.run(method_key=method, on_progress=_progress_callback(cfg.rounds))
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s (device={result.get('device', '?')})", file=sys.stderr)

    # Print summary
    summary = _extract_summary(result)
    print(f"\n{'=' * 50}")
    print(f"  Paradigm: {cfg.paradigm.upper()}")
    print(f"  Method: {method}")
    if cfg.name:
        print(f"  Name: {cfg.name}")
    print(f"  Dataset: {cfg.dataset} | Partition: {cfg.partition}")
    print(f"  Rounds: {cfg.rounds} | Clients: {cfg.total_clients} (K={cfg.clients_per_round})")
    if cfg.paradigm == "fd":
        print(f"  Public dataset: {cfg.public_dataset} ({cfg.public_dataset_size} samples)")
        if cfg.model_heterogeneous:
            print(f"  Model pool: {cfg.model_pool}")
        if cfg.channel_noise:
            print(f"  Channel: UL={cfg.ul_snr_db}dB DL={cfg.dl_snr_db}dB BS={cfg.n_bs_antennas}")
        if cfg.group_based:
            print(f"  Group-based: FedTSKD-G (threshold={cfg.channel_threshold})")
    print(f"{'=' * 50}")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")
    print(f"{'=' * 50}")

    # Save results — use run_dir as primary location
    run_dir = Path(str(sim.run_dir))
    out_path = args.output
    if out_path is None:
        out_path = str(run_dir / "results.json")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nRun directory: {run_dir}")
    print(f"Results saved to: {out_path}")

    sim.cleanup()


def cmd_compare(args):
    """Compare multiple methods on the same partition."""
    cfg = _args_to_config(args)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if not methods:
        print("Error: No methods specified. Use --methods 'method1,method2'", file=sys.stderr)
        sys.exit(1)

    paradigm_str = f" [{cfg.paradigm.upper()}]" if cfg.paradigm != "fl" else ""
    name_str = f" (name={cfg.name})" if cfg.name else ""
    print(f"Comparing{paradigm_str} {len(methods)} methods on {cfg.dataset} ({cfg.partition}), "
          f"{cfg.rounds} rounds, {cfg.total_clients} clients (K={cfg.clients_per_round}){name_str}", file=sys.stderr)

    # Create ONE simulator with shared partition
    sim = _create_simulator(cfg)
    sim.setup()

    results = {}
    for method in methods:
        print(f"\n--- Running: {method} ---", file=sys.stderr)
        t0 = time.time()
        result = sim.run(method_key=method, on_progress=_progress_callback(cfg.rounds))
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s", file=sys.stderr)
        results[method] = result

    # Print comparison table
    headers = ["Method", "Accuracy", "Loss", "F1", "Gini", "Wall Clock", "TFLOPs", "Comm"]
    rows = []
    for method in methods:
        s = _extract_summary(results[method])
        rows.append([method, s["accuracy"], s["loss"], s["f1"],
                      s["fairness_gini"], s["wall_clock"], s["cum_tflops"], s["cum_comm"]])

    print(f"\n{'=' * 80}")
    print(f"  Comparison: {cfg.dataset} | {cfg.partition} | {cfg.rounds} rounds")
    print(f"{'=' * 80}")
    _print_table(headers, rows)
    print(f"{'=' * 80}")

    # Save all results to the run directory
    run_dir = Path(str(sim.run_dir))
    out_path = args.output
    if out_path is None:
        out_path = str(run_dir / "compare_results.json")

    payload = {"config": asdict(cfg), "methods": methods, "results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nRun directory: {run_dir}")
    print(f"Results saved to: {out_path}")

    sim.cleanup()


def cmd_plot(args):
    """Generate publication-quality plots from a completed run."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for file output
    import matplotlib.pyplot as plt

    fmt = args.format
    dpi = args.dpi
    plot_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    style = args.style

    # Resolve run(s) — can be a single run or a compare results file
    run_path = _resolve_run_dir(args.run)
    data = _load_run_results(run_path)

    # Determine output directory
    out_dir = Path(args.out_dir) if args.out_dir else (run_path / "plots" if run_path.is_dir() else Path("."))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect if this is a compare result (has "results" dict with multiple methods)
    # or a single run (has "metrics" list)
    is_compare = "results" in data and isinstance(data["results"], dict)

    if is_compare:
        method_metrics = {}  # {method: {metric: [values]}}
        for method_key, method_result in data["results"].items():
            metrics_list = method_result.get("metrics", [])
            method_metrics[method_key] = {}
            for metric in plot_metrics:
                values = []
                for row in metrics_list:
                    r = row.get("round", -1)
                    if isinstance(r, (int, float)) and r >= 0:
                        values.append(float(row.get(metric, 0.0) or 0.0))
                method_metrics[method_key][metric] = values
    else:
        metrics_list = data.get("metrics", [])
        single_name = data.get("method", "run")
        method_metrics = {single_name: {}}
        for metric in plot_metrics:
            values = []
            for row in metrics_list:
                r = row.get("round", -1)
                if isinstance(r, (int, float)) and r >= 0:
                    values.append(float(row.get(metric, 0.0) or 0.0))
            method_metrics[single_name][metric] = values

    methods = list(method_metrics.keys())

    # Build short display names for legend (strip category prefix, prettify)
    _short = {}
    _pretty = {
        "fedavg": "FedAvg", "random": "Random", "delta": "DELTA",
        "fedcs": "FedCS", "oort": "Oort", "oort_plus": "Oort+",
        "ucb_grad": "UCB-Grad", "apex": "APEX",
        "apex_no_phase": "APEX-noPhase", "apex_no_ts": "APEX-noTS",
        "apex_no_div": "APEX-noDiv", "tifl": "TiFL", "poc": "PoC",
        "epsilon_greedy": "e-Greedy", "linucb": "LinUCB",
        "fedcor_approx": "FedCor", "fedcor": "FedCor",
        "apex_v2": "APEX v2",
        "apex_v2_no_adaptive_recency": "v2-noRecency",
        "apex_v2_no_hysteresis": "v2-noHyst",
        "apex_v2_no_het_scaling": "v2-noHetScale",
        "apex_v2_no_posterior_reg": "v2-noPosterior",
        "apex_v2_no_adaptive_gamma": "v2-noAdaGamma",
    }
    for m in methods:
        short = m.split(".")[-1] if "." in m else m
        _short[m] = _pretty.get(short, short)

    # IEEE-recommended matplotlib settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "text.usetex": False,  # set True if LaTeX is available
    })

    saved_files = []

    # --- Individual metric plots ---
    for metric in plot_metrics:
        with plt.style.context(style if style in plt.style.available else "default"):
            fig, ax = plt.subplots(figsize=(args.width, args.height))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            for method in methods:
                ys = method_metrics[method].get(metric, [])
                xs = list(range(len(ys)))
                ax.plot(xs, ys, label=_short[method], linewidth=1.5)

            ax.set_xlabel("Round")
            ax.set_ylabel(_metric_label(metric))
            ax.set_title(_metric_label(metric) + " per Round")

            if len(methods) > 1:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                          fontsize=6, framealpha=0.9,
                          handlelength=1.2, handletextpad=0.4,
                          borderaxespad=0.3)

            fig.tight_layout(rect=[0.0, 0.0, 0.75, 1.0])

            filename = f"{metric}.{fmt}"
            filepath = out_dir / filename
            fig.savefig(str(filepath), format=fmt, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(filepath)

    # --- Multi-panel plot (if multiple metrics) ---
    if len(plot_metrics) >= 2:
        n_panels = min(len(plot_metrics), 4)
        n_cols = 2 if n_panels > 1 else 1
        n_rows = (n_panels + n_cols - 1) // n_cols

        with plt.style.context(style if style in plt.style.available else "default"):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(args.width * n_cols, args.height * n_rows))
            fig.patch.set_facecolor("white")
            if n_panels == 1:
                axes = [[axes]]
            elif n_rows == 1:
                axes = [axes]

            for idx, metric in enumerate(plot_metrics[:n_panels]):
                r, c = divmod(idx, n_cols)
                ax = axes[r][c]
                ax.set_facecolor("white")
                for method in methods:
                    ys = method_metrics[method].get(metric, [])
                    xs = list(range(len(ys)))
                    ax.plot(xs, ys, label=_short[method], linewidth=1.5)
                ax.set_xlabel("Round")
                ax.set_ylabel(_metric_label(metric))
                ax.set_title(_metric_label(metric))

            # Hide unused panels
            for idx in range(n_panels, n_rows * n_cols):
                r, c = divmod(idx, n_cols)
                axes[r][c].set_visible(False)

            # Shared legend to the right of all panels
            handles, labels = axes[0][0].get_legend_handles_labels()
            if len(methods) > 1:
                fig.legend(handles, labels, loc="center left",
                           bbox_to_anchor=(1.02, 0.5),
                           framealpha=0.9, fontsize=6, handlelength=1.2,
                           handletextpad=0.4, borderaxespad=0.3)

            fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
            filename = f"multi_panel.{fmt}"
            filepath = out_dir / filename
            fig.savefig(str(filepath), format=fmt, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(filepath)

    # Print results
    print(f"Saved {len(saved_files)} plot(s) to: {out_dir}")
    for f in saved_files:
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    print(f"\nFormat: {fmt.upper()} | DPI: {dpi} | Style: {style}")


def _metric_label(key: str) -> str:
    """Convert metric key to a human-readable label."""
    labels = {
        "accuracy": "Accuracy",
        "loss": "Loss",
        "f1": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
        "wall_clock": "Wall Clock (s)",
        "round_time": "Round Time (s)",
        "cum_tflops": "Cumulative TFLOPs",
        "cum_comm": "Communication (MB)",
        "fairness_var": "Fairness Variance",
        "fairness_gini": "Fairness (Gini)",
        "composite": "Composite Score",
        "dp_used_avg": "DP Epsilon Used (avg)",
        "training_tflops": "Training TFLOPs",
        "selection_time": "Selection Time (s)",
        "compute_time": "Compute Time (s)",
        "round_energy": "Round Energy",
        "round_bytes": "Round Bytes",
        # FD-specific metrics
        "kl_divergence_avg": "KL Divergence (avg)",
        "distillation_loss_avg": "Distillation Loss (avg)",
        "logit_comm_kb": "Logit Communication (KB)",
        "fl_equiv_comm_mb": "FL Equiv. Communication (MB)",
        "comm_reduction_ratio": "Comm. Reduction Ratio",
        "effective_noise_var": "Effective Noise Variance",
        "dynamic_steps_kr": "Dynamic Steps K_r",
        "num_good_channel": "Good Channel Clients",
        "num_bad_channel": "Bad Channel Clients",
        "client_accuracy_avg": "Client Accuracy (avg)",
        "client_accuracy_std": "Client Accuracy (std)",
    }
    return labels.get(key, key.replace("_", " ").title())


def cmd_list_methods(args):
    """List all registered selection methods."""
    reg = MethodRegistry()
    reg.load_presets()

    headers = ["Key", "Display Name", "Origin"]
    rows = []
    for key in sorted(reg.methods.keys()):
        name = reg.display_names.get(key, key)
        origin = reg.origins.get(key, "unknown")
        rows.append([key, name, origin])

    _print_table(headers, rows)
    print(f"\nTotal: {len(rows)} methods")


def cmd_list_runs(args):
    """List all saved runs with their names and key metrics."""
    from csfl_simulator.core.utils import ART_ROOT
    runs_dir = ART_ROOT / "runs"
    if not runs_dir.is_dir():
        print("No runs found.")
        return

    headers = ["Name", "Method", "Dataset", "Rounds", "Accuracy", "Path"]
    rows = []

    for run_dir in sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        results_path = run_dir / "results.json"
        compare_path = run_dir / "compare_results.json"

        method = "?"
        dataset = "?"
        rounds = "?"
        accuracy = "?"

        # Try loading config
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                c = cfg.get("config", cfg)
                dataset = c.get("dataset", "?")
            except Exception:
                pass

        # Try loading results (single run)
        if results_path.exists():
            try:
                with open(results_path) as f:
                    res = json.load(f)
                method = res.get("method", "?")
                metrics = res.get("metrics", [])
                if metrics:
                    final = metrics[-1]
                    rounds = str(final.get("round", "?"))
                    accuracy = f"{final.get('accuracy', 0.0):.4f}"
            except Exception:
                pass
        elif compare_path.exists():
            try:
                with open(compare_path) as f:
                    res = json.load(f)
                method_list = res.get("methods", [])
                method = ",".join(method_list[:3])
                if len(method_list) > 3:
                    method += f"...+{len(method_list) - 3}"
                # Show best accuracy across methods
                best_acc = 0.0
                for m, r in res.get("results", {}).items():
                    ms = r.get("metrics", [])
                    if ms:
                        best_acc = max(best_acc, ms[-1].get("accuracy", 0.0))
                        rounds = str(ms[-1].get("round", "?"))
                accuracy = f"{best_acc:.4f}"
            except Exception:
                pass
        elif metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    mdata = json.load(f)
                metrics = mdata.get("metrics", [])
                if metrics:
                    final = metrics[-1]
                    rounds = str(final.get("round", "?"))
                    accuracy = f"{final.get('accuracy', 0.0):.4f}"
            except Exception:
                pass

        rows.append([run_dir.name, method, dataset, rounds, accuracy, str(run_dir)])

    if rows:
        _print_table(headers, rows)
        print(f"\nTotal: {len(rows)} run(s)")
    else:
        print("No runs found.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="csfl_simulator",
        description="CSFL Simulator - Client Selection for Federated Learning",
    )
    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="Run a single simulation")
    _add_sim_args(run_p)
    run_p.add_argument("--method", required=True, help="Selection method key (use list-methods to see all)")

    # compare
    cmp_p = sub.add_parser("compare", help="Compare multiple methods on the same partition")
    _add_sim_args(cmp_p)
    cmp_p.add_argument("--methods", required=True, help="Comma-separated method keys")

    # plot
    plot_p = sub.add_parser("plot", help="Generate publication-quality plots (EPS/PDF/PNG) from a run")
    plot_p.add_argument("--run", required=True,
                        help="Run directory, path to results JSON, or name prefix (e.g. 'exp1')")
    plot_p.add_argument("--metrics", default="accuracy,loss,f1",
                        help="Comma-separated metrics to plot (default: accuracy,loss,f1)")
    plot_p.add_argument("--format", default="eps", choices=["eps", "pdf", "png", "svg"],
                        help="Output format (default: eps for IEEE)")
    plot_p.add_argument("--dpi", type=int, default=300, help="Resolution (default: 300)")
    plot_p.add_argument("--style", default="classic",
                        help="Matplotlib style (classic, default, seaborn-v0_8, etc.)")
    plot_p.add_argument("--width", type=float, default=3.5,
                        help="Figure width in inches (default: 3.5 = IEEE single column)")
    plot_p.add_argument("--height", type=float, default=2.6,
                        help="Figure height in inches (default: 2.6)")
    plot_p.add_argument("--out-dir", default=None,
                        help="Output directory (default: <run_dir>/plots/)")

    # list-methods
    sub.add_parser("list-methods", help="List all registered selection methods")

    # list-runs
    sub.add_parser("list-runs", help="List all saved runs")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "list-methods":
        cmd_list_methods(args)
    elif args.command == "list-runs":
        cmd_list_runs(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
