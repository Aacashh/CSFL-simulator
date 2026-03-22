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
    p.add_argument("--fast-mode", action="store_true", default=True, help="Fast mode (break after 2 batches)")
    p.add_argument("--no-fast-mode", dest="fast_mode", action="store_false", help="Disable fast mode")
    p.add_argument("--time-budget", type=float, default=None, help="Time budget per round")
    p.add_argument("--energy-budget", type=float, default=None, help="Energy budget per round")
    p.add_argument("--bytes-budget", type=float, default=None, help="Bytes budget per round")
    p.add_argument("--track-grad-norm", action="store_true", default=False, help="Track gradient norms")
    p.add_argument("--parallel-clients", type=int, default=0, help="Parallel client training (0=off, -1=auto)")
    p.add_argument("--output", "-o", default=None, help="Output JSON path (auto-generated if not specified)")
    p.add_argument("--dp-sigma", type=float, default=0.0, help="DP Gaussian noise sigma")
    p.add_argument("--dp-epsilon-per-round", type=float, default=0.0, help="DP epsilon budget per round")
    p.add_argument("--dp-clip-norm", type=float, default=0.0, help="DP gradient clip norm")


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
        fast_mode=args.fast_mode,
        time_budget=args.time_budget,
        energy_budget=args.energy_budget,
        bytes_budget=args.bytes_budget,
        track_grad_norm=args.track_grad_norm,
        parallel_clients=args.parallel_clients,
        dp_sigma=args.dp_sigma,
        dp_epsilon_per_round=args.dp_epsilon_per_round,
        dp_clip_norm=args.dp_clip_norm,
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
    return {
        "accuracy": f"{final.get('accuracy', 0.0):.4f}",
        "loss": f"{final.get('loss', 0.0):.4f}",
        "f1": f"{final.get('f1', 0.0):.4f}",
        "fairness_gini": f"{final.get('fairness_gini', 0.0):.4f}",
        "wall_clock": f"{final.get('wall_clock', 0.0):.2f}s",
        "cum_tflops": f"{final.get('cum_tflops', 0.0):.6f}",
        "cum_comm": f"{final.get('cum_comm', 0.0):.2f}MB",
    }


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

def cmd_run(args):
    """Run a single simulation."""
    cfg = _args_to_config(args)
    method = args.method

    name_str = f" (name={cfg.name})" if cfg.name else ""
    print(f"Running: method={method} dataset={cfg.dataset} partition={cfg.partition} "
          f"clients={cfg.total_clients} K={cfg.clients_per_round} rounds={cfg.rounds}{name_str}", file=sys.stderr)

    sim = FLSimulator(cfg)
    t0 = time.time()
    result = sim.run(method_key=method, on_progress=_progress_callback(cfg.rounds))
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s (device={result.get('device', '?')})", file=sys.stderr)

    # Print summary
    summary = _extract_summary(result)
    print(f"\n{'=' * 50}")
    print(f"  Method: {method}")
    if cfg.name:
        print(f"  Name: {cfg.name}")
    print(f"  Dataset: {cfg.dataset} | Partition: {cfg.partition}")
    print(f"  Rounds: {cfg.rounds} | Clients: {cfg.total_clients} (K={cfg.clients_per_round})")
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

    name_str = f" (name={cfg.name})" if cfg.name else ""
    print(f"Comparing {len(methods)} methods on {cfg.dataset} ({cfg.partition}), "
          f"{cfg.rounds} rounds, {cfg.total_clients} clients (K={cfg.clients_per_round}){name_str}", file=sys.stderr)

    # Create ONE simulator with shared partition
    sim = FLSimulator(cfg)
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
                ncol = min(len(methods), 4)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                          ncol=ncol, fontsize=6, framealpha=0.9,
                          handlelength=1.2, columnspacing=0.6, handletextpad=0.4)

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.28 if len(methods) > 4 else 0.22)

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

            # Shared legend below all panels
            handles, labels = axes[0][0].get_legend_handles_labels()
            if len(methods) > 1:
                fig.legend(handles, labels, loc="upper center",
                           ncol=min(len(methods), 7), bbox_to_anchor=(0.5, -0.02),
                           framealpha=0.9, fontsize=6, handlelength=1.2,
                           columnspacing=0.6, handletextpad=0.4)

            fig.tight_layout()
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
