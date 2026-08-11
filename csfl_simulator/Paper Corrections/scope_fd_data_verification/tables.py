from load import load, agg
import collections

rows = [r for r in load() if not r["incomplete"]]


def show(fam, label):
    sub = [r for r in rows if r["family"] == fam]
    print("=== %s (%s) ===" % (label, fam))
    if sub:
        c = sub[0]
        print("    config: N=%s K=%s R=%s alpha=%s dataset=%s | seeds=%s"
              % (c["N"], c["K"], c["R"], c["alpha"], c["dataset"],
                 sorted(set(r["seed"] for r in sub))))
    by = collections.defaultdict(list)
    for r in sub:
        by[r["method"]].append(r)
    for m in sorted(by):
        v = by[m]
        a, asd, n = agg([x["acc"] * 100 for x in v])
        g, gsd, _ = agg([x["gini"] * 100 for x in v])
        r70, r70sd, n70 = agg([x["r70"] for x in v])
        print("  %-36s acc=%6.2f +- %4.2f  gini=%6.2f +- %.4f  r70=%s  n=%d"
              % (m, a, asd, g, gsd,
                 ("%.1f" % r70) if r70 is not None else "  n/a", n))
    print()


show("ablation_headline", "TABLE II  ablation")
show("literature_baselines", "TABLE III  selector comparison")
