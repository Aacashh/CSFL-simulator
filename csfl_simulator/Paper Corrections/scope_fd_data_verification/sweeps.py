from load import load, agg
import collections

rows = [r for r in load() if not r["incomplete"]]
S = "fd_native.scope_fd"; D = "fd_native.scope_fd_debt_only"; RND = "heuristic.random"


def by(fam, key):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["family"] == fam:
            out[key(r)][r["method"]].append(r)
    return out


def line(v):
    a, sd, n = agg([x["acc"] * 100 for x in v])
    g, _, _ = agg([x["gini"] * 100 for x in v])
    return "acc=%6.2f+-%4.2f gini=%6.2f n=%d" % (a, sd, g, n)


print("########## DIRICHLET SWEEP (paper Sec VI-E) ##########")
for k, m in sorted(by("dirichlet_severity", lambda r: r["alpha"]).items(),
                   key=lambda kv: (kv[0] is None, kv[0])):
    print(" alpha=%-6s" % k)
    for meth in [S, RND]:
        if m.get(meth):
            print("    %-28s %s" % (meth, line(m[meth])))

print()
print("########## K SWEEP (paper Sec VI-F) ##########")
for k, m in sorted(by("ablation_k_sweep", lambda r: r["K"]).items()):
    print(" K=%-4s" % k)
    for meth in [S, D]:
        if m.get(meth):
            print("    %-28s %s" % (meth, line(m[meth])))
for k, m in sorted(by("literature_baselines_k_sweep", lambda r: r["K"]).items()):
    print(" K=%-4s (literature_baselines_k_sweep)" % k)
    for meth in [S, RND]:
        if m.get(meth):
            print("    %-28s %s" % (meth, line(m[meth])))

print()
print("########## IID SANITY ##########")
for k, m in sorted(by("iid_sanity", lambda r: r["alpha"]).items(),
                   key=lambda kv: (kv[0] is None, kv[0])):
    for meth in sorted(m):
        print("  %-30s %s" % (meth, line(m[meth])))
