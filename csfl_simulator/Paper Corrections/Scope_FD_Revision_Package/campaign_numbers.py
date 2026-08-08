#!/usr/bin/env python3
"""Regenerate every campaign-derived number quoted in the SCOPE-FD revision.

Reads only compare_results.json, which the runner writes once a job finishes, so
a partially written run cannot enter a mean.  Run this again whenever new runs
land and diff the output against the previous copy.

    python3 campaign_numbers.py [runs-root] > campaign_numbers.txt

Sections
    1  campaign inventory and completeness
    2  exact law for the participation Gini coefficient
    3  headline, ablation and ported-selector tables
    4  accuracy gap against the participation ratio
    5  cross-domain results
    6  invariance census
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "runs_scope_revised" / "runs_scope_revised"
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT

SCOPE = "fd_native.scope_fd"
DEBT = "fd_native.scope_fd_debt_only"
RANDOM = "heuristic.random"
NAME = {
    RANDOM: "uniform random", SCOPE: "SCOPE-FD", DEBT: "debt only",
    "fd_native.scope_fd_no_diversity": "debt + under-prediction",
    "fd_native.scope_fd_no_server": "debt + coverage",
    "fd_native.divfl_fd": "DivFL", "fd_native.subtrunc_fd": "SubTrunc",
    "fd_native.unionfl_fd": "UnionFL", "system_aware.oort": "Oort",
    "fd_native.scope_fd_channel_aware": "channel aware",
    "fd_native.scope_fd_surrogate_hist": "surrogate histogram",
}
for eps in ("0_1", "0_5", "1", "2", "5"):
    NAME[f"fd_native.scope_fd_hist_dp_eps{eps}"] = "Laplace eps=" + eps.replace("_", ".")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def runs():
    for res in sorted(ROOT.rglob("compare_results.json")):
        man = res.parent / "manifest.json"
        manifest = json.loads(man.read_text()) if man.is_file() else {}
        payload = json.loads(res.read_text())
        family = manifest.get("family") or res.parent.parent.name
        yield family, res.parent.relative_to(ROOT), payload.get("config", {}), payload.get("results", {})


def finals(result):
    rows = [x for x in result.get("metrics", []) if int(x.get("round", -1)) >= 0]
    return rows[-1] if rows else None


def at_round(result, r):
    for x in result.get("metrics", []):
        if int(x.get("round", -1)) == r:
            return x
    return None


def mean_sd(vals):
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None, 0
    return (statistics.fmean(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0.0,
            len(vals))


def fmt(vals, scale=100.0, digits=2):
    m, s, n = mean_sd(vals)
    if m is None:
        return "--"
    return f"{m*scale:.{digits}f} +- {s*scale:.{digits}f}  (n={n})"


def signtest(k, n):
    """Two-sided exact sign test for k successes in n trials."""
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def wilcox(diffs):
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    d = [x for x in diffs if abs(x) > 1e-15]
    if len(d) < 2:
        return None
    return float(wilcoxon(d, alternative="two-sided").pvalue)


def head(n, title):
    print()
    print("=" * 78)
    print(f"{n}.  {title.upper()}")
    print("=" * 78)


# --------------------------------------------------------------------------
ALL = list(runs())


def s1_inventory():
    head(1, "campaign inventory")
    fams = defaultdict(list)
    for fam, tag, cfg, results in ALL:
        fams[fam].append((tag, cfg, results))
    total_cells = 0
    print(f"{'family':<32}{'runs':>6}{'seeds':>8}{'methods':>9}")
    for fam in sorted(fams):
        rs = fams[fam]
        seeds = sorted({int(c.get("seed", -1)) for _, c, _ in rs})
        methods = sorted({m for _, _, r in rs for m in r})
        total_cells += len(rs)
        print(f"{fam:<32}{len(rs):>6}{len(seeds):>8}{len(methods):>9}")
    print(f"{'TOTAL':<32}{total_cells:>6}")
    print(f"\n{len(fams)} families, {total_cells} completed runs")

    incomplete = []
    for d in sorted(ROOT.glob("*/*")):
        if d.is_dir() and not (d / "compare_results.json").is_file():
            incomplete.append(d.relative_to(ROOT))
    print(f"{len(incomplete)} run directories without a result file:")
    for d in incomplete:
        print(f"    {d}")


def s2_gini_law():
    head(2, "exact law for the participation Gini coefficient")
    print("Claim  G(R) = m(N-m)/(N K R)  with  m = KR mod N,  for any selector whose")
    print("       counts differ by at most one.  Maximum over m is N/(4KR) at m = N/2,")
    print("       and G = 0 exactly when N divides KR.\n")
    # The law describes the intended rotation.  Two things break that link and
    # are therefore held out: client dropout, where a selected client may fail
    # to return, and the channel-aware variant, whose energy filter can leave a
    # round short of K clients.  A plain-selector run that merely carries an
    # energy budget is kept, because the plain score reads no energy term.
    ok = bad = skipped = 0
    per_method = defaultdict(int)
    worst = 0.0
    table = {}
    for fam, tag, cfg, results in ALL:
        N, K, R = cfg.get("total_clients"), cfg.get("clients_per_round"), cfg.get("rounds")
        if not (N and K and R):
            continue
        if cfg.get("dropout_prob"):
            skipped += sum(1 for m in (SCOPE, DEBT) if m in results)
            continue
        for method in (SCOPE, DEBT):
            r = results.get(method)
            f = finals(r) if r else None
            if not f or f.get("fairness_gini") is None:
                continue
            m = (K * R) % N
            pred = m * (N - m) / (N * K * R)
            err = abs(float(f["fairness_gini"]) - pred)
            worst = max(worst, err)
            if err < 5e-4:
                ok += 1
                per_method[method] += 1
            else:
                bad += 1
                print(f"  MISMATCH {fam}/{tag} {method} N={N} K={K} R={R} "
                      f"measured={f['fairness_gini']*100:.3f} predicted={pred*100:.3f}")
            table[(N, K, R)] = (float(f["fairness_gini"]), pred)
    print(f"{ok} selector runs agree with the law, {bad} disagree, "
          f"largest deviation {worst*100:.4f} pp")
    for m, c in sorted(per_method.items()):
        print(f"    {NAME.get(m, m):<24}{c:>5} runs")
    print(f"{skipped} runs held out: dropout changes which clients actually participated\n")
    print(f"{'N':>5}{'K':>5}{'R':>5}{'K|N':>6}{'N|KR':>7}{'KR mod N':>10}"
          f"{'measured %':>12}{'law %':>9}{'bound %':>9}")
    for (N, K, R) in sorted(table):
        meas, pred = table[(N, K, R)]
        print(f"{N:>5}{K:>5}{R:>5}{('yes' if N % K == 0 else 'no'):>6}"
              f"{('yes' if (K*R) % N == 0 else 'no'):>7}{(K*R) % N:>10}"
              f"{meas*100:>12.3f}{pred*100:>9.3f}{100*N/(4*K*R):>9.3f}")

    print("\nHorizon slice at the headline N=30 K=5, five seeds")
    sc, rn = defaultdict(list), defaultdict(list)
    for fam, tag, cfg, results in ALL:
        if fam != "literature_baselines":
            continue
        for R in (25, 50, 75, 100):
            for meth, store in ((SCOPE, sc), (RANDOM, rn)):
                row = at_round(results.get(meth, {}), R - 1)
                if row and row.get("fairness_gini") is not None:
                    store[R].append(row["fairness_gini"])
    print(f"{'R':>5}{'law %':>9}{'SCOPE-FD measured':>22}{'uniform random':>24}")
    for R in (25, 50, 75, 100):
        m = (5 * R) % 30
        print(f"{R:>5}{m*(30-m)/(30*5*R)*100:>9.3f}{fmt(sc[R]):>22}{fmt(rn[R]):>24}")


def cell(family=None, **want):
    """Collect method -> seed -> final row for runs matching a config filter."""
    out = defaultdict(dict)
    for fam, tag, cfg, results in ALL:
        if family and fam != family:
            continue
        if any(cfg.get(k) != v for k, v in want.items()):
            continue
        seed = int(cfg.get("seed", -1))
        for method, r in results.items():
            f = finals(r)
            if f:
                out[method][seed] = f
    return out


def show(c, methods=None, key="accuracy"):
    methods = methods or sorted(c, key=lambda m: -statistics.fmean(
        v[key] for v in c[m].values()))
    ref = c.get(SCOPE, {})
    print(f"   {'selector':<24}{'accuracy %':>20}{'Gini %':>18}{'rolling Gini %':>20}{'p':>8}")
    for m in methods:
        if m not in c:
            continue
        acc = [v["accuracy"] for v in c[m].values()]
        gin = [v.get("fairness_gini") for v in c[m].values()]
        rol = [v.get("rolling_window_gini") for v in c[m].values()]
        p = ""
        if m != SCOPE and ref:
            seeds = sorted(set(ref) & set(c[m]))
            pv = wilcox([(ref[s]["accuracy"] - c[m][s]["accuracy"]) * 100 for s in seeds])
            p = f"{pv:.3f}" if pv is not None else ""
        print(f"   {NAME.get(m, m):<24}{fmt(acc):>20}{fmt(gin):>18}{fmt(rol):>20}{p:>8}")


def s3_tables():
    head(3, "headline, ablation and ported selectors")
    print("Four-way ablation, Fashion-MNIST N=30 K=5 alpha=0.5")
    show(cell(family="ablation_headline"))
    print("\nPorted selectors, Fashion-MNIST N=30 K=5 alpha=0.5")
    show(cell(family="literature_baselines"))
    print("\nHistogram privacy, Fashion-MNIST N=30 K=5 alpha=0.5")
    show(cell(family="histogram_privacy"))
    print("\nDirichlet sweep")
    for a in (0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 5.0):
        c = cell(family="dirichlet_severity", dirichlet_alpha=a)
        if c:
            print(f"\n  alpha = {a}")
            show(c, [SCOPE, DEBT, RANDOM, "fd_native.divfl_fd", "fd_native.subtrunc_fd"])
    print("\nPublic-set sensitivity")
    for label, want in (
            ("public set = MNIST", dict(public_dataset="MNIST")),
            ("public set size 500", dict(public_dataset_size=500)),
            ("public set size 100", dict(public_dataset_size=100)),
            ("public label noise 0.1", dict(public_label_noise=0.1)),
            ("public label noise 0.3", dict(public_label_noise=0.3))):
        c = cell(family="public_dataset_sensitivity", **want)
        if c:
            print(f"\n  {label}")
            show(c, [SCOPE, DEBT, RANDOM])


def s4_ratio():
    head(4, "accuracy gap against the participation ratio K/N")
    print("Fashion-MNIST, alpha=0.5, no dropout, no staleness, no budget, no channel noise,")
    print("public set drawn from the held-out split at 2000 samples.\n")
    cells = defaultdict(lambda: defaultdict(dict))
    for fam, tag, cfg, results in ALL:
        if RANDOM not in results:
            continue
        if (cfg.get("dropout_prob") or cfg.get("staleness_window")
                or cfg.get("energy_budget") or cfg.get("channel_noise")):
            continue
        if cfg.get("dataset") != "Fashion-MNIST" or cfg.get("dirichlet_alpha") != 0.5:
            continue
        if (cfg.get("public_dataset") != "same" or cfg.get("public_dataset_size") != 2000
                or cfg.get("public_label_noise")):
            continue
        key = (cfg["total_clients"], cfg["clients_per_round"])
        seed = int(cfg.get("seed", -1))
        for m in (SCOPE, DEBT, RANDOM):
            f = finals(results.get(m, {}))
            if f:
                cells[key][m][seed] = f["accuracy"]

    print(f"{'N':>5}{'K':>4}{'K/N':>8}{'n':>4}"
          f"{'SCOPE-FD gap':>15}{'wins':>7}{'p':>7}"
          f"{'debt-only gap':>16}{'wins':>7}{'p':>7}")
    scope_rows, debt_rows, ratios = [], [], []
    for key in sorted(cells, key=lambda k: k[1] / k[0]):
        N, K = key
        c = cells[key]
        if SCOPE not in c or RANDOM not in c:
            continue
        line = f"{N:>5}{K:>4}{K/N:>8.3f}"
        n_used = 0
        vals = {}
        for m in (SCOPE, DEBT):
            if m not in c:
                vals[m] = ("--", "--", "")
                continue
            seeds = sorted(set(c[m]) & set(c[RANDOM]))
            gaps = [(c[m][s] - c[RANDOM][s]) * 100 for s in seeds]
            n_used = len(seeds)
            pv = wilcox(gaps)
            vals[m] = (f"{statistics.fmean(gaps):+.2f}",
                       f"{sum(1 for g in gaps if g > 0)}/{len(gaps)}",
                       f"{pv:.3f}" if pv is not None else "")
            if m == SCOPE:
                scope_rows.append(statistics.fmean(gaps))
            else:
                debt_rows.append(statistics.fmean(gaps))
        ratios.append(K / N)
        print(line + f"{n_used:>4}"
              + f"{vals[SCOPE][0]:>15}{vals[SCOPE][1]:>7}{vals[SCOPE][2]:>7}"
              + f"{vals.get(DEBT, ('--','--',''))[0]:>16}"
              + f"{vals.get(DEBT, ('--','--',''))[1]:>7}"
              + f"{vals.get(DEBT, ('--','--',''))[2]:>7}")

    print()
    for label, rows in (("SCOPE-FD", scope_rows), ("debt only", debt_rows)):
        pos = sum(1 for g in rows if g > 0)
        print(f"{label:>10}: {len(rows)} configurations, {pos} with a positive mean gap, "
              f"range {min(rows):+.2f} to {max(rows):+.2f} pp, "
              f"sign test p = {signtest(pos, len(rows)):.5f}")
        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(ratios[:len(rows)], rows)
            print(f"{'':>10}  Spearman rho against K/N = {rho:+.3f}, p = {p:.5f}")
        except Exception:
            pass


def s5_cross_domain():
    head(5, "cross-domain results")
    print("Free Spoken Digit Dataset, five seeds, three local epochs, 150-sample public set")
    show(cell(family="audio_fsdd", local_epochs=3))
    print("\nEarlier audio attempt, three seeds, one local epoch, 300-sample public set")
    show(cell(family="audio_fsdd", local_epochs=1))
    print("\nMNIST private, three seeds")
    show(cell(family="dataset_generality", dataset="MNIST"))
    print("\nCIFAR-10 private with STL-10 public")
    c = cell(family="cifar10_multiseed")
    show(c)
    seeds = sorted({s for m in c.values() for s in m})
    print(f"   seeds completed: {seeds}")
    print("\nPublic set disjoint from the evaluation data")
    print("   Two configurations draw the public set from a separate corpus:")
    print("   MNIST public with Fashion-MNIST private, and STL-10 public with CIFAR-10 private.")


def s6_invariance():
    head(6, "invariance census for the participation Gini coefficient")
    counts = defaultdict(set)
    total = 0
    for fam, tag, cfg, results in ALL:
        f = finals(results.get(SCOPE, {}))
        if not f or f.get("fairness_gini") is None:
            continue
        total += 1
        counts[round(float(f["fairness_gini"]) * 100, 2)].add(fam)
    per_value = defaultdict(int)
    for fam, tag, cfg, results in ALL:
        f = finals(results.get(SCOPE, {}))
        if f and f.get("fairness_gini") is not None:
            per_value[round(float(f["fairness_gini"]) * 100, 2)] += 1
    print(f"{total} runs include the proposed selector\n")
    print(f"{'Gini %':>9}{'runs':>7}{'families':>10}   families")
    for v in sorted(per_value, key=lambda x: -per_value[x]):
        print(f"{v:>9.2f}{per_value[v]:>7}{len(counts[v]):>10}   "
              + ", ".join(sorted(counts[v])))
    all_fams = {fam for fam, _, _, _ in ALL}
    print(f"\ntotal families in the campaign: {len(all_fams)}")


if __name__ == "__main__":
    print(f"SCOPE-FD campaign numbers")
    print(f"source: {ROOT}")
    s1_inventory()
    s2_gini_law()
    s3_tables()
    s4_ratio()
    s5_cross_domain()
    s6_invariance()
