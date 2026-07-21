"""Aggregate matched multi-seed SCOPE-FD runs into mean/std/95% CI tables."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

DEFAULT_METRICS = (
    "accuracy",
    "loss",
    "f1",
    "fairness_gini",
    "rolling_window_gini",
    "server_accuracy",
    "wall_clock",
)


def summarize(values: Iterable[float]) -> Dict[str, float | int | None]:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    n = len(vals)
    if not vals:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if n > 1 else 0.0
    if n > 1:
        try:
            from scipy.stats import t
            critical = float(t.ppf(0.975, n - 1))
        except Exception:
            critical = 1.96
        half = critical * std / math.sqrt(n)
    else:
        half = 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def rounds_to_threshold(metrics: list[dict], threshold: float):
    for row in metrics:
        if int(row.get("round", -1)) >= 0 and float(row.get("accuracy", 0.0) or 0.0) >= threshold:
            return int(row["round"]) + 1
    return None


def _result_files(paths: list[Path]):
    seen = set()
    for path in paths:
        candidates = [path] if path.is_file() else path.rglob("compare_results.json")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def _load_record(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = path.with_name("manifest.json")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    config = payload.get("config", {})
    family = manifest.get("family", "ungrouped")
    group_config = {
        key: value
        for key, value in config.items()
        if key not in {"seed", "name"}
    }
    group_key = json.dumps(
        {"family": family, "config": group_config}, sort_keys=True, separators=(",", ":")
    )
    return group_key, family, group_config, payload


def _paired_test(reference: Dict[int, float], baseline: Dict[int, float]):
    seeds = sorted(set(reference) & set(baseline))
    if len(seeds) < 2:
        return {"n": len(seeds), "p_value": None}
    x = [reference[seed] for seed in seeds]
    y = [baseline[seed] for seed in seeds]
    if all(abs(a - b) <= 1e-15 for a, b in zip(x, y)):
        return {"n": len(seeds), "p_value": 1.0}
    try:
        from scipy.stats import wilcoxon
        test = wilcoxon(x, y, alternative="two-sided")
        return {"n": len(seeds), "p_value": float(test.pvalue)}
    except Exception:
        return {"n": len(seeds), "p_value": None}


def aggregate(paths: list[Path], reference_method: str):
    grouped: dict[str, dict] = {}
    seed_values = defaultdict(lambda: defaultdict(dict))

    for path in _result_files(paths):
        group_key, family, config, payload = _load_record(path)
        group = grouped.setdefault(
            group_key,
            {"family": family, "config": config, "methods": defaultdict(lambda: {"runs": []})},
        )
        seed = int(payload.get("config", {}).get("seed", -1))
        for method, result in payload.get("results", {}).items():
            metrics = result.get("metrics", [])
            rounds = [row for row in metrics if int(row.get("round", -1)) >= 0]
            if not rounds:
                continue
            final = rounds[-1]
            convergence = dict(result.get("convergence", {}))
            for threshold in (0.6, 0.7, 0.8):
                convergence.setdefault(
                    f"rounds_to_abs_{int(threshold * 100)}",
                    rounds_to_threshold(metrics, threshold),
                )
            group["methods"][method]["runs"].append(
                {
                    "seed": seed,
                    "final": final,
                    "convergence": convergence,
                    "metrics": rounds,
                    "source": str(path),
                }
            )
            seed_values[group_key][method][seed] = float(final.get("accuracy", 0.0))

    output_groups = {}
    for group_key, group in grouped.items():
        methods_out = {}
        for method, method_data in group["methods"].items():
            runs = method_data["runs"]
            metric_names = set(DEFAULT_METRICS)
            metric_names.update(
                key for run in runs for key in run["final"] if isinstance(run["final"][key], (int, float))
            )
            final_summary = {
                metric: summarize(run["final"].get(metric) for run in runs)
                for metric in sorted(metric_names)
            }
            convergence_names = set(
                key for run in runs for key in run["convergence"]
                if isinstance(run["convergence"][key], (int, float)) or run["convergence"][key] is None
            )
            convergence_summary = {
                metric: summarize(run["convergence"].get(metric) for run in runs)
                for metric in sorted(convergence_names)
            }

            curve_rounds = sorted(
                set(int(row["round"]) for run in runs for row in run["metrics"])
            )
            curves = {}
            for metric in DEFAULT_METRICS:
                curves[metric] = []
                for round_idx in curve_rounds:
                    vals = [
                        row.get(metric)
                        for run in runs
                        for row in run["metrics"]
                        if int(row["round"]) == round_idx
                    ]
                    curves[metric].append({"round": round_idx, **summarize(vals)})
            methods_out[method] = {
                "seeds": sorted(run["seed"] for run in runs),
                "final": final_summary,
                "convergence": convergence_summary,
                "curves": curves,
            }

        comparisons = {}
        reference = seed_values[group_key].get(reference_method, {})
        for method, values in seed_values[group_key].items():
            if method != reference_method and reference:
                comparisons[method] = _paired_test(reference, values)

        output_groups[group_key] = {
            "family": group["family"],
            "config": group["config"],
            "methods": methods_out,
            "reference_method": reference_method,
            "paired_wilcoxon_final_accuracy": comparisons,
        }
    return {"schema_version": 1, "groups": output_groups}


def _latex_table(aggregated: dict) -> str:
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Family & Method & Accuracy & Gini & Rounds to 70\% \\",
        r"\hline",
    ]
    for group in aggregated["groups"].values():
        family = str(group["family"]).replace("_", r"\_")
        for method, data in group["methods"].items():
            method_tex = method.replace("_", r"\_")
            acc = data["final"].get("accuracy", {})
            gini = data["final"].get("fairness_gini", {})
            r70 = data["convergence"].get("rounds_to_abs_70", {})
            def cell(summary):
                if summary.get("mean") is None:
                    return "--"
                return f"{summary['mean']:.4f} $\\pm$ {summary['std']:.4f}"
            lines.append(
                f"{family} & {method_tex} & {cell(acc)} & {cell(gini)} & {cell(r70)} \\\\"
            )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs_scope_revised/aggregated"))
    parser.add_argument("--reference-method", default="fd_native.scope_fd")
    args = parser.parse_args()

    aggregated = aggregate(args.runs, args.reference_method)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "aggregated_results.json"
    json_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    (args.output_dir / "summary_table.tex").write_text(
        _latex_table(aggregated), encoding="utf-8"
    )
    print(f"Wrote {json_path} ({len(aggregated['groups'])} configuration groups)")


if __name__ == "__main__":
    main()
