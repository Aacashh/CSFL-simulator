"""Pool every distinct MAML-Select run and recompute coverage and Jain's index.

Section V-C and Reply 4 both quote a pooled figure over every MAML-Select run in
the study, with and without the forced cold start. That pool grows whenever a
run is added, so the count and the three Jain pairs have to be recomputed rather
than carried forward. This is the script that settles what those numbers are.

The pool is every run whose method is MAML-Select, deduplicated by output
directory, since the same run can be reachable under more than one path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import fairness
import load

# Which dataset a run belongs to, read from its own configuration where
# possible and from the run label otherwise.
DATASETS = [("fashion", "Fashion-MNIST"), ("cifar100", "CIFAR-100"),
            ("cifar10", "CIFAR-10")]


def dataset_of(run):
    name = (run["config"].get("dataset") or "").lower()
    hay = (name + " " + run["label"] + " " + run["scenario"] + " "
           + run["experiment"]).lower()
    # cifar100 has to be tested before cifar10, since one contains the other
    for key, label in DATASETS:
        if key in hay:
            return label
    return "unknown"


def is_maml(run):
    """Every MAML-Select run, including the ablation variants.

    The ablations carry keys like research.maml_select.no_battery, so matching
    on the display name alone finds only the six unmodified runs and misses the
    sixty the pooled claim is actually about.
    """
    return run["method_key"].startswith(("research.maml_select",
                                         "ml.maml_select"))


def main():
    runs = [r for r in load.load_all() if is_maml(r)]

    seen, pool = set(), []
    for r in runs:
        d = os.path.dirname(r["path"])
        if d in seen:
            continue
        seen.add(d)
        pool.append(r)

    print("distinct MAML-Select runs found: %d" % len(pool))
    print()

    rows = {}
    for r in pool:
        rows.setdefault(dataset_of(r), []).append(r)

    print("%-14s %5s %10s %10s %9s %9s" %
          ("dataset", "runs", "cov all", "cov post", "Jain all", "Jain post"))
    print("-" * 62)
    total = 0
    for label in ("Fashion-MNIST", "CIFAR-10", "CIFAR-100", "unknown"):
        rs = rows.get(label)
        if not rs:
            continue
        s = fairness.summarise(rs)
        total += len(rs)
        # every run has to reach full coverage for the claim to hold as stated
        worst_all = min(x["cov_all"] for x in s)
        worst_post = min(x["cov_post"] for x in s)
        print("%-14s %5d %10.1f %10.1f %9.3f %9.3f" %
              (label, len(rs), worst_all, worst_post,
               sum(x["jain_all"] for x in s) / len(s),
               sum(x["jain_post"] for x in s) / len(s)))
        below = [x["label"] for x in s if x["cov_post"] < 100.0]
        if below:
            print("      runs below full post-warmup coverage: %s" % below[:5])
    print("-" * 62)
    print("%-14s %5d" % ("total", total))
    print()
    print("Coverage columns are the worst single run, not a mean, because the")
    print("claim is that coverage is complete in every individual run.")


main()
