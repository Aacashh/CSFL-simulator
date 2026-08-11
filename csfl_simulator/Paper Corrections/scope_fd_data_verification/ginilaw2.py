from load import load
import collections

rows = [r for r in load() if not r["incomplete"]]
SEL = {"fd_native.scope_fd", "fd_native.scope_fd_debt_only"}

# Held out: dropout (a selected client may fail to return, so realised
# participation is not intended participation) and the channel-aware variant
# (the energy budget filter makes clients unselectable). The plain arms inside
# channel_energy are NOT held out -- they are unaffected by the budget.
def held_out(r):
    return r["family"] == "dropout"

ok = bad = 0
byfam = collections.Counter(); vals = collections.Counter()
permeth = collections.Counter(); famvals = collections.defaultdict(set)
for r in rows:
    if r["method"] not in SEL or r["gini"] is None or held_out(r):
        continue
    N, K, R = r["N"], r["K"], r["R"]
    m = (K * R) % N
    pred = m * (N - m) / float(N * K * R)
    if abs(pred - r["gini"]) < 1e-12:
        ok += 1; byfam[r["family"]] += 1; permeth[r["method"]] += 1
        vals[round(r["gini"] * 100, 4)] += 1
        famvals[round(r["gini"] * 100, 4)].add(r["family"])
    else:
        bad += 1
        print("MISMATCH", r["family"], r["method"], N, K, R, pred, r["gini"])

print("EXACT Gini law holds: %d / %d   (zero deviation, not approximate)" % (ok, ok + bad))
print("  scope_fd arms      :", permeth["fd_native.scope_fd"])
print("  debt_only arms     :", permeth["fd_native.scope_fd_debt_only"])
print("  families           :", len(byfam))
heldn = sum(1 for r in rows if r["method"] in SEL and held_out(r))
print("  held out (dropout) :", heldn)
print()
print("distinct Gini values returned and how often:")
for v, c in sorted(vals.items()):
    print("   %7.4f%%  x%-4d  across %d families" % (v, c, len(famvals[v])))
print()
print("per-family:")
for f, c in sorted(byfam.items()):
    print("   %-32s %d" % (f, c))
