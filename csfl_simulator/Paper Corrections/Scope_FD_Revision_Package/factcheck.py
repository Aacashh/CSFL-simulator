#!/usr/bin/env python3
"""Check every number quoted in the manuscript against the run data.

Two passes.

  Pass A  Extracts every "mean +- sd" pair and every bare percentage from the
          experimental sections and searches the campaign for a cell that
          produces it. Anything with no source is either stale, mistyped, or
          computed from data that is not on this machine, and all three need
          looking at.

  Pass B  Asserts the specific structural claims that are not a single number,
          such as the closed form for the Gini coefficient, the rotation
          partition property, and the rank correlation against K/N.

    python3 factcheck.py [manuscript.tex] [runs-root]

Exit status is non-zero if any assertion in pass B fails.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "main_scope_revised.tex"
ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[2] / "runs_scope_revised" / "runs_scope_revised"

SCOPE = "fd_native.scope_fd"
DEBT = "fd_native.scope_fd_debt_only"
RANDOM = "heuristic.random"
TOL = 0.006          # values are quoted to two decimals

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------- load runs
def load():
    out = []
    for res in sorted(ROOT.rglob("compare_results.json")):
        man = res.parent / "manifest.json"
        manifest = json.loads(man.read_text()) if man.is_file() else {}
        payload = json.loads(res.read_text())
        out.append((manifest.get("family") or res.parent.parent.name,
                    res.parent.relative_to(ROOT), payload.get("config", {}),
                    payload.get("results", {})))
    return out


ALL = load()

CELL_KEYS = ("dataset", "public_dataset", "public_dataset_size", "public_label_noise",
             "dirichlet_alpha", "total_clients", "clients_per_round", "local_epochs",
             "dl_snr_db", "channel_noise", "energy_budget", "dropout_prob",
             "staleness_window", "rounds")


def finals(result):
    rows = [x for x in result.get("metrics", []) if int(x.get("round", -1)) >= 0]
    return rows[-1] if rows else None


def at_round(result, r):
    for x in result.get("metrics", []):
        if int(x.get("round", -1)) == r:
            return x
    return None


def build_cells():
    cells = defaultdict(lambda: defaultdict(dict))
    for fam, tag, cfg, results in ALL:
        key = (fam,) + tuple(cfg.get(k) for k in CELL_KEYS)
        seed = int(cfg.get("seed", -1))
        for method, r in results.items():
            f = finals(r)
            if f:
                cells[key][method][seed] = f
    return cells


CELLS = build_cells()


def stat(vals):
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return (statistics.fmean(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0.0, len(vals))


def catalogue():
    """Every (mean, sd) and every mean the campaign can produce, with its source."""
    pairs, singles = defaultdict(list), defaultdict(list)
    for key, methods in CELLS.items():
        fam = key[0]
        for method, runs in methods.items():
            label = f"{fam}/{method.split('.')[-1]} n={len(runs)}"
            for field, scale in (("accuracy", 100.0), ("fairness_gini", 100.0),
                                 ("rolling_window_gini", 100.0), ("f1", 100.0)):
                s = stat(r.get(field) for r in runs.values())
                if s is None:
                    continue
                m, sd, n = s
                pairs[(round(m * scale, 2), round(sd * scale, 2))].append(f"{label} {field}")
                singles[round(m * scale, 2)].append(f"{label} {field}")
    # Horizon slices, which the paper quotes at R = 25, 50, 75, 100. Both the
    # per-run value and the mean over seeds within a family are catalogued,
    # since the paper quotes the multi-seed mean.
    slices = defaultdict(list)
    for fam, tag, cfg, results in ALL:
        for method, r in results.items():
            for R in (25, 50, 75, 100):
                row = at_round(r, R - 1)
                if row and row.get("fairness_gini") is not None:
                    singles[round(row["fairness_gini"] * 100, 2)].append(
                        f"{fam}/{method.split('.')[-1]} gini@R={R}")
                    slices[(fam, method, R)].append(row["fairness_gini"])
    for (fam, method, R), vals in slices.items():
        singles[round(statistics.fmean(vals) * 100, 2)].append(
            f"{fam}/{method.split('.')[-1]} mean gini@R={R} n={len(vals)}")

    # The coefficient grid varies alpha_u and alpha_d, which the run config does
    # not carry. They live in the manifest, so that family is grouped here.
    grid = defaultdict(list)
    for d in sorted((ROOT / "coefficient_grid").glob("*")):
        rp, mp = d / "compare_results.json", d / "manifest.json"
        if not (rp.is_file() and mp.is_file()):
            continue
        cfg = json.loads(mp.read_text()).get("config", {})
        f = finals(json.loads(rp.read_text())["results"].get(SCOPE, {}))
        if f and "scope_au" in cfg:
            grid[(cfg["scope_au"], cfg["scope_ad"])].append(f["accuracy"])
    for (au, ad), vals in grid.items():
        singles[round(statistics.fmean(vals) * 100, 2)].append(
            f"coefficient_grid cell au={au} ad={ad} n={len(vals)}")

    # Values pooled across the cells of a family, which the paper does for the
    # channel-aware variant across the three SNR settings.
    pooled = defaultdict(list)
    for fam, tag, cfg, results in ALL:
        for method, r in results.items():
            f = finals(r)
            if f and f.get("fairness_gini") is not None:
                pooled[(fam, method)].append(f["fairness_gini"])
    for (fam, method), vals in pooled.items():
        singles[round(statistics.fmean(vals) * 100, 2)].append(
            f"{fam}/{method.split('.')[-1]} pooled gini n={len(vals)}")

    # The analytic bound N/(4KR), which is derived rather than measured.
    for N, K, R in {(cfg.get("total_clients"), cfg.get("clients_per_round"), cfg.get("rounds"))
                    for _, _, cfg, _ in ALL if cfg.get("total_clients")}:
        for RR in (25, 50, 75, R):
            singles[round(100 * N / (4 * K * RR), 2)].append(
                f"analytic bound N/(4KR) at N={N} K={K} R={RR}")
    return pairs, singles


PAIRS, SINGLES = catalogue()


def near(table, value, tol=TOL):
    hits = []
    for k, v in table.items():
        if isinstance(k, tuple):
            if abs(k[0] - value[0]) <= tol and abs(k[1] - value[1]) <= tol:
                hits += v
        elif abs(k - value) <= tol:
            hits += v
    return hits


# ---------------------------------------------------------------- pass A
def pass_a():
    print("=" * 78)
    print("PASS A   every quoted number traced to a run")
    print("=" * 78)
    src = TEX.read_text()
    body = src.split(r"\section{Experimental Results}", 1)[1]
    body = body.split(r"\section{Conclusion", 1)[0]

    pm = sorted({(float(a), float(b)) for a, b in
                 re.findall(r"\$(\d+\.\d+)\s*\\pm\s*(\d+\.\d+)\$", body)})
    pct = sorted({float(x) for x in re.findall(r"\$(\d+\.\d+)\\%\$", body)})

    unsourced = []
    print(f"\n{len(pm)} distinct mean +- sd pairs")
    for value in pm:
        hits = near(PAIRS, value)
        if not hits:
            unsourced.append(f"{value[0]:.2f} +- {value[1]:.2f}")
        else:
            print(f"  ok   {value[0]:>6.2f} +- {value[1]:<5.2f}  <-  {hits[0]}"
                  + (f"  (+{len(hits)-1} more)" if len(hits) > 1 else ""))
    for u in unsourced:
        print(f"  ??   {u:<20} NO SOURCE FOUND")

    print(f"\n{len(pct)} distinct bare percentages")
    unsourced_pct = []
    for value in pct:
        hits = near(SINGLES, value)
        if not hits:
            unsourced_pct.append(value)
        else:
            print(f"  ok   {value:>6.2f}%  <-  {hits[0]}"
                  + (f"  (+{len(hits)-1} more)" if len(hits) > 1 else ""))
    for u in unsourced_pct:
        print(f"  ??   {u:>6.2f}%  no direct match, check by hand")

    print(f"\nsummary: {len(pm)-len(unsourced)}/{len(pm)} pairs and "
          f"{len(pct)-len(unsourced_pct)}/{len(pct)} percentages traced")
    return unsourced, unsourced_pct


# ---------------------------------------------------------------- pass B
def pass_b():
    print()
    print("=" * 78)
    print("PASS B   structural claims")
    print("=" * 78)

    # 1. the closed form for the Gini coefficient
    ok = bad = 0
    for fam, tag, cfg, results in ALL:
        N, K, R = cfg.get("total_clients"), cfg.get("clients_per_round"), cfg.get("rounds")
        if not (N and K and R) or cfg.get("dropout_prob"):
            continue
        for method in (SCOPE, DEBT):
            f = finals(results.get(method, {}))
            if not f or f.get("fairness_gini") is None:
                continue
            m = (K * R) % N
            ok += abs(f["fairness_gini"] - m * (N - m) / (N * K * R)) < 5e-4
            bad += abs(f["fairness_gini"] - m * (N - m) / (N * K * R)) >= 5e-4
    check(bad == 0 and ok == 457, "Gini closed form holds in 457 of 457 selector runs",
          f"{ok} agree, {bad} disagree")

    # 2. zero exactly when N divides KR, in both directions
    z_when = all(
        (finals(res.get(SCOPE, {})) or {}).get("fairness_gini", 1) < 5e-4
        for _, _, cfg, res in ALL
        if not cfg.get("dropout_prob") and cfg.get("total_clients")
        and (cfg["clients_per_round"] * cfg["rounds"]) % cfg["total_clients"] == 0
        and SCOPE in res)
    nz_when = all(
        (finals(res.get(SCOPE, {})) or {}).get("fairness_gini", 0) > 5e-4
        for _, _, cfg, res in ALL
        if not cfg.get("dropout_prob") and cfg.get("total_clients")
        and (cfg["clients_per_round"] * cfg["rounds"]) % cfg["total_clients"] != 0
        and SCOPE in res)
    check(z_when and nz_when, "Gini is zero exactly when N divides KR")

    # counterexamples the paper names
    def gini_at(N, K):
        for _, _, cfg, res in ALL:
            if (cfg.get("total_clients") == N and cfg.get("clients_per_round") == K
                    and not cfg.get("dropout_prob") and SCOPE in res):
                f = finals(res[SCOPE])
                if f:
                    return f["fairness_gini"] * 100
        return None
    check(abs(gini_at(30, 5) - 1.333) < 0.01, "N=30 K=5 has K|N yet Gini is 1.33 percent",
          f"measured {gini_at(30,5):.3f}")
    check(gini_at(50, 3) is not None and gini_at(50, 3) < 0.01,
          "N=50 K=3 has K not dividing N yet Gini is zero",
          f"measured {gini_at(50,3):.3f}")

    # 3. rotation partitions every aligned cycle window at N=30 K=10
    tot = good = 0
    cohort_counts = {SCOPE: [], DEBT: []}
    for fam, tag, cfg, results in ALL:
        if cfg.get("total_clients") != 30 or cfg.get("clients_per_round") != 10:
            continue
        for method in (SCOPE, DEBT):
            sel = (results.get(method) or {}).get("history", {}).get("selected")
            if not sel:
                continue
            cohort_counts[method].append(len({frozenset(s) for s in sel}))
            for w in range(len(sel) // 3):
                tot += 1
                good += sorted(set(sel[3*w]) | set(sel[3*w+1]) | set(sel[3*w+2])) == list(range(30))
    check(tot > 0 and good == tot, "every aligned 3-round window partitions the pool",
          f"{good}/{tot} windows")
    lo, hi = min(cohort_counts[SCOPE]), max(cohort_counts[SCOPE])
    check((lo, hi) == (35, 54), "complete score uses 35 to 54 distinct cohorts", f"{lo} to {hi}")
    check(set(cohort_counts[DEBT]) == {3}, "debt-only uses exactly 3 distinct cohorts",
          str(sorted(set(cohort_counts[DEBT]))))

    # 4. participation counts at N=30 K=10
    rnd_sd, rnd_lo, rnd_hi, sc_sd = [], [], [], []
    for fam, tag, cfg, results in ALL:
        if cfg.get("total_clients") != 30 or cfg.get("clients_per_round") != 10:
            continue
        for method, store in ((RANDOM, rnd_sd), (SCOPE, sc_sd)):
            pc = (results.get(method) or {}).get("participation_counts")
            if not pc:
                continue
            v = sorted(pc.values()) if isinstance(pc, dict) else sorted(pc)
            store.append(statistics.pstdev(v))
            if method == RANDOM:
                rnd_lo.append(min(v)); rnd_hi.append(max(v))
    check(abs(statistics.fmean(rnd_sd) - 4.53) < 0.01, "uniform random count sd is 4.53",
          f"{statistics.fmean(rnd_sd):.3f} over {len(rnd_sd)} seeds")
    check((min(rnd_lo), max(rnd_hi)) == (23, 42), "uniform random counts span 23 to 42",
          f"{min(rnd_lo)} to {max(rnd_hi)}")
    check(all(abs(s - 0.4714) < 0.001 for s in sc_sd), "proposed selector count sd is 0.47",
          f"{sorted(set(round(s,4) for s in sc_sd))}")
    check(abs(statistics.fmean(rnd_sd) / statistics.fmean(sc_sd) - 9.6) < 0.1,
          "count sd reduction is about 9.6x",
          f"{statistics.fmean(rnd_sd)/statistics.fmean(sc_sd):.2f}x")

    # 5. the K/N ratio result
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
        for method in (SCOPE, DEBT, RANDOM):
            f = finals(results.get(method, {}))
            if f:
                cells[(cfg["total_clients"], cfg["clients_per_round"])][method][
                    int(cfg["seed"])] = f["accuracy"]
    ratios, sgap, dgap = [], [], []
    for (N, K), c in cells.items():
        if SCOPE not in c or RANDOM not in c:
            continue
        seeds = sorted(set(c[SCOPE]) & set(c[RANDOM]))
        ratios.append(K / N)
        sgap.append(statistics.fmean((c[SCOPE][s] - c[RANDOM][s]) * 100 for s in seeds))
        ds = sorted(set(c.get(DEBT, {})) & set(c[RANDOM]))
        dgap.append(statistics.fmean((c[DEBT][s] - c[RANDOM][s]) * 100 for s in ds) if ds else None)
    check(len(ratios) == 15, "15 matched configurations", f"{len(ratios)}")
    three = sorted((N, K) for (N, K), c in cells.items()
                   if SCOPE in c and len(set(c[SCOPE]) & set(c[RANDOM])) == 3)
    five = [(N, K) for (N, K), c in cells.items()
            if SCOPE in c and len(set(c[SCOPE]) & set(c[RANDOM])) == 5]
    check(three == [(30, 1), (30, 3), (30, 10)] and len(five) == 12,
          "three seeds at 30/1, 30/3, 30/10 and five everywhere else",
          f"three-seed {three}, five-seed count {len(five)}")
    check(abs(min(ratios) - 1/30) < 1e-9 and abs(max(ratios) - 1/3) < 1e-9,
          "ratios span 0.033 to 0.333", f"{min(ratios):.3f} to {max(ratios):.3f}")
    check(sum(1 for g in sgap if g > 0) == 12, "complete score positive in 12 of 15",
          f"{sum(1 for g in sgap if g > 0)}")
    check(all(g > 0 for g in dgap), "debt-only positive in 15 of 15",
          f"{sum(1 for g in dgap if g > 0)} of {len(dgap)}")
    check(abs(max(sgap) - 2.43) < 0.01 and abs(min(sgap) + 1.42) < 0.01,
          "complete score gap spans -1.42 to +2.43 pp",
          f"{min(sgap):+.2f} to {max(sgap):+.2f}")
    check(abs(min(dgap) - 0.05) < 0.01 and abs(max(dgap) - 2.39) < 0.01,
          "debt-only gap spans +0.05 to +2.39 pp",
          f"{min(dgap):+.2f} to {max(dgap):+.2f}")
    try:
        from scipy.stats import spearmanr
        rs, ps = spearmanr(ratios, sgap)
        rd, pd = spearmanr(ratios, dgap)
        check(abs(rs + 0.815) < 0.005 and ps < 0.0005,
              "complete score Spearman -0.815 at p = 0.0002", f"rho={rs:.3f} p={ps:.5f}")
        check(abs(rd + 0.764) < 0.005 and pd < 0.001,
              "debt-only Spearman -0.764 at p = 0.0009", f"rho={rd:.3f} p={pd:.5f}")
    except ImportError:
        print("  SKIP  Spearman correlations, scipy not available")

    # the three configurations where the complete score trails
    trail = sorted((N, K) for (N, K), c in cells.items()
                   if SCOPE in c and RANDOM in c
                   and statistics.fmean(
                       (c[SCOPE][s] - c[RANDOM][s]) * 100
                       for s in sorted(set(c[SCOPE]) & set(c[RANDOM]))) < 0)
    check(trail == [(100, 20), (200, 20), (200, 40)],
          "the three trailing configurations are N=100/K=20, 200/20, 200/40", str(trail))
    check(all(K >= 20 for _, K in trail), "all trailing configurations have K >= 20")
    margins = []
    for N, K in trail:
        c = cells[(N, K)]
        s = sorted(set(c[SCOPE]) & set(c[DEBT]))
        margins.append(round(statistics.fmean((c[DEBT][x] - c[SCOPE][x]) * 100 for x in s), 2))
    check(margins == [0.67, 1.34, 1.90], "debt-only leads by 0.67, 1.34, 1.90 pp there",
          str(margins))

    # 6. audio, five seeds, every seed a win over random
    aud = None
    for key, methods in CELLS.items():
        if key[0] == "audio_fsdd" and key[1 + CELL_KEYS.index("local_epochs")] == 3:
            aud = methods
    seeds = sorted(set(aud[SCOPE]) & set(aud[RANDOM]))
    gaps = [(aud[SCOPE][s]["accuracy"] - aud[RANDOM][s]["accuracy"]) * 100 for s in seeds]
    check(len(seeds) == 5 and all(g > 0 for g in gaps),
          "audio: proposed selector wins on all five seeds", f"{sorted(round(g,2) for g in gaps)}")
    check(abs(min(gaps) - 0.51) < 0.01 and abs(max(gaps) - 1.70) < 0.01,
          "audio: per-seed margin 0.51 to 1.70 pp", f"{min(gaps):.2f} to {max(gaps):.2f}")
    # The manuscript says "at least four" rather than "more than five" because
    # SubTrunc on seed 11 gives 4.02, which is the binding case.
    for other in ("fd_native.divfl_fd", "fd_native.subtrunc_fd"):
        g = [(aud[SCOPE][s]["accuracy"] - aud[other][s]["accuracy"]) * 100 for s in seeds]
        check(all(x >= 4 for x in g),
              f"audio: beats {other.split('.')[-1]} by at least 4 pp on every seed",
              f"min {min(g):.2f}")

    # 7. campaign totals quoted in the invariance section
    check(len(ALL) == 284, "284 completed runs", str(len(ALL)))
    check(len({f for f, _, _, _ in ALL}) == 17, "17 families",
          str(len({f for f, _, _, _ in ALL})))
    n133 = sum(1 for _, _, cfg, res in ALL
               if (finals(res.get(SCOPE, {})) or {}).get("fairness_gini") is not None
               and abs(finals(res[SCOPE])["fairness_gini"] - 0.013333) < 5e-5)
    fam133 = {f for f, _, cfg, res in ALL
              if (finals(res.get(SCOPE, {})) or {}).get("fairness_gini") is not None
              and abs(finals(res[SCOPE])["fairness_gini"] - 0.013333) < 5e-5}
    check(n133 == 202, "1.33 percent returned in 202 runs", str(n133))
    check(len(fam133) == 15, "spanning 15 of the 17 families", str(len(fam133)))
    ndrop = sum(1 for _, _, cfg, res in ALL if cfg.get("dropout_prob") and SCOPE in res)
    check(ndrop == 9, "9 dropout runs of the proposed selector", str(ndrop))
    dg = [finals(res[SCOPE])["fairness_gini"] * 100 for _, _, cfg, res in ALL
          if cfg.get("dropout_prob") and SCOPE in res]
    check(abs(min(dg) - 0.42) < 0.01 and abs(max(dg) - 2.12) < 0.01,
          "dropout moves the coefficient over 0.42 to 2.12 percent",
          f"{min(dg):.2f} to {max(dg):.2f}")

    # 8. the coefficient grid
    grid = defaultdict(list)
    for d in sorted((ROOT / "coefficient_grid").glob("*")):
        rp, mp = d / "compare_results.json", d / "manifest.json"
        if not (rp.is_file() and mp.is_file()):
            continue
        cfg = json.loads(mp.read_text()).get("config", {})
        f = finals(json.loads(rp.read_text())["results"].get(SCOPE, {}))
        if f and "scope_au" in cfg:
            grid[(cfg["scope_au"], cfg["scope_ad"])].append(f)
    means = {k: statistics.fmean(x["accuracy"] * 100 for x in v) for k, v in grid.items()}
    check(len(means) == 30, "coefficient grid has 30 cells", str(len(means)))
    check(all(len(v) == 3 for v in grid.values()), "3 seeds in every grid cell")
    check(abs(min(means.values()) - 70.94) < 0.01 and abs(max(means.values()) - 72.27) < 0.01,
          "grid accuracy spans 70.94 to 72.27",
          f"{min(means.values()):.2f} to {max(means.values()):.2f}")
    check(abs(max(means.values()) - min(means.values()) - 1.33) < 0.01,
          "grid spread is 1.33 pp",
          f"{max(means.values())-min(means.values()):.2f}")
    check(abs(means[(0.3, 0.1)] - 71.99) < 0.01, "the chosen cell (0.3, 0.1) reaches 71.99",
          f"{means[(0.3,0.1)]:.2f}")
    check(abs(max(means.values()) - means[(0.3, 0.1)] - 0.27) < 0.01,
          "the chosen cell is 0.27 pp off the best",
          f"{max(means.values())-means[(0.3,0.1)]:.2f}")
    gv = {round(x["fairness_gini"] * 100, 3) for v in grid.values() for x in v}
    check(gv == {1.333}, "Gini is 1.33 percent in all 90 grid runs", str(sorted(gv)))

    # 9. the channel-aware variant, pooled over the three SNR settings
    ca = [finals(res["fd_native.scope_fd_channel_aware"])["fairness_gini"] * 100
          for fam, _, cfg, res in ALL
          if fam == "channel_energy" and "fd_native.scope_fd_channel_aware" in res]
    check(len(ca) == 9 and abs(statistics.fmean(ca) - 11.69) < 0.01,
          "channel-aware Gini is 11.69 percent pooled over 9 runs",
          f"{statistics.fmean(ca):.2f} over {len(ca)} runs")

    # 10. no single-seed result is quoted anywhere in the results sections
    body = TEX.read_text().split(r"\section{Experimental Results}", 1)[1]
    body = body.split(r"\section{Conclusion", 1)[0]
    for phrase in ("single seed", "one seed", "a single random seed"):
        check(phrase not in body.lower(), f"results section does not mention {phrase!r}")


if __name__ == "__main__":
    print(f"manuscript : {TEX}")
    print(f"runs       : {ROOT}\n")
    pass_a()
    pass_b()
    print()
    print("=" * 78)
    if failures:
        print(f"{len(failures)} STRUCTURAL CHECKS FAILED")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
    print("all structural checks passed")
