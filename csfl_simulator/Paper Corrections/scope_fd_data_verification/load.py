"""Load every completed SCOPE-FD run into a flat table of method-arms."""
import json, os, glob, math

ROOT = r"C:/Users/drash/OneDrive/Desktop/CSFL-simulator/runs/runs_scope_revised"


def gini(counts):
    n = len(counts)
    tot = sum(counts)
    if tot == 0:
        return 0.0
    num = sum(abs(a - b) for a in counts for b in counts)
    return num / (2.0 * n * tot)


def load():
    rows = []
    for mf in glob.glob(os.path.join(ROOT, "*", "*", "manifest.json")):
        rundir = os.path.dirname(mf)
        family = os.path.basename(os.path.dirname(rundir))
        man = json.load(open(mf))
        cr = os.path.join(rundir, "compare_results.json")
        # Older orchestrator versions wrote no "status" key at all; those runs
        # are still valid if they produced a results file. Only "failed" and a
        # missing results file mean the run did not finish.
        if man.get("status") == "failed" or not os.path.exists(cr):
            rows.append(dict(family=family, hash=man.get("hash"), status=man.get("status"),
                             method=None, incomplete=True))
            continue
        try:
            d = json.load(open(cr))
        except Exception as e:
            print("BAD JSON", cr, e)
            continue
        cfg = man.get("config") or man.get("resolved_config") or d.get("config", {})
        for method, r in d.get("results", {}).items():
            pc = r.get("participation_counts") or []
            conv = r.get("convergence") or {}
            rows.append(dict(
                family=family, hash=man.get("hash") or os.path.basename(rundir), method=method, incomplete=False,
                seed=cfg.get("seed"), N=cfg.get("total_clients"), K=cfg.get("clients_per_round"),
                R=cfg.get("rounds"), alpha=cfg.get("dirichlet_alpha"),
                dataset=cfg.get("dataset"), public=cfg.get("public_dataset"),
                public_size=cfg.get("public_dataset_size"),
                local_epochs=cfg.get("local_epochs"),
                acc=conv.get("final_accuracy"),
                r70=conv.get("rounds_to_abs_70"), r60=conv.get("rounds_to_abs_60"),
                r80=conv.get("rounds_to_abs_80"),
                gini=gini(pc) if pc else None, counts=pc,
                cfg=cfg,
            ))
    return rows


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0, len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, sd, len(vals)


if __name__ == "__main__":
    rows = load()
    ok = [r for r in rows if not r["incomplete"]]
    bad = [r for r in rows if r["incomplete"]]
    print("method-arms loaded:", len(ok))
    print("incomplete run dirs:", len(bad))
    fams = sorted(set(r["family"] for r in ok))
    print("families:", len(fams))
    for f in fams:
        sub = [r for r in ok if r["family"] == f]
        print("  %-32s arms=%3d  runs=%3d" % (f, len(sub), len(set(x["hash"] for x in sub))))
