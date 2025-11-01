from __future__ import annotations
import argparse
from pathlib import Path
import json
import csv

from csfl_simulator.core.utils import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=str((ROOT / "artifacts" / "runs").resolve()))
    ap.add_argument("--output", default=str((ROOT / "artifacts" / "exports" / "metrics.csv").resolve()))
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in sorted(runs_dir.glob("sim_*")):
        mpath = run / "metrics.json"
        cpath = run / "config.json"
        try:
            metr = json.loads(mpath.read_text())
            cfg = json.loads(cpath.read_text())
        except Exception:
            continue
        metrics = metr.get("metrics", [])
        method = cfg.get("config", {}).get("method", None)
        if method is None:
            # attempt to infer from directory or skip
            method = "unknown"
        for m in metrics:
            rows.append({
                "run_id": run.name,
                "round": int(m.get("round", -1)),
                "accuracy": float(m.get("accuracy", 0.0)),
                "round_time": float(m.get("round_time", 0.0)),
                "fairness_var": float(m.get("fairness_var", 0.0)),
                "dp_used_avg": float(m.get("dp_used_avg", 0.0)),
                "composite": float(m.get("composite", 0.0)),
                "method": method,
            })

    if not rows:
        print("No rows to export.")
        return
    # Write CSV
    keys = ["run_id", "method", "round", "accuracy", "round_time", "fairness_var", "dp_used_avg", "composite"]
    with out_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"Exported {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()


