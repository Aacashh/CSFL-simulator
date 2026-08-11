from load import load, agg
import collections
rows = [r for r in load() if not r["incomplete"]]
S="fd_native.scope_fd"; D="fd_native.scope_fd_debt_only"; RND="heuristic.random"

def grp(fam, key):
    o=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["family"]==fam: o[key(r)][r["method"]].append(r)
    return o
def L(v):
    a,sd,n=agg([x["acc"]*100 for x in v]); g,_,_=agg([x["gini"]*100 for x in v])
    return "acc=%6.2f+-%4.2f gini=%6.2f n=%d"%(a,sd,g,n)

print("########## SCALE / NON-DIVISIBLE (Sec VI-L) ##########")
for k,m in sorted(grp("scale_and_nondivisible", lambda r:(r["N"],r["K"])).items()):
    star = "*" if k[0]%k[1] else " "
    parts=[]
    for meth,tag in [(S,"scope"),(D,"debt"),(RND,"rand")]:
        if m.get(meth):
            a,sd,n=agg([x["acc"]*100 for x in m[meth]]); g,_,_=agg([x["gini"]*100 for x in m[meth]])
            parts.append("%s %5.2f/%5.2f"%(tag,a,g))
    print("  N=%-4d K=%-3d %s %s"%(k[0],k[1],star," | ".join(parts)))

print()
print("########## CROSS-DATASET (Sec VI-M) ##########")
for fam in ["dataset_generality","audio_fsdd","cifar10_multiseed"]:
    print(" ---",fam)
    for k,m in sorted(grp(fam, lambda r:(r["dataset"],r["N"],r["K"])).items()):
        print("   %s"%(k,))
        for meth in sorted(m): print("      %-40s %s"%(meth,L(m[meth])))

print()
print("########## PRIVACY (Sec VI-I) ##########")
for k,m in sorted(grp("histogram_privacy", lambda r:0).items()):
    for meth in sorted(m): print("   %-44s %s"%(meth,L(m[meth])))

print()
print("########## DROPOUT / STALENESS (Sec VI-J) ##########")
for fam in ["dropout","bounded_staleness"]:
    print(" ---",fam)
    for k,m in sorted(grp(fam, lambda r:r["hash"]).items()):
        pass
    for meth in sorted(set(r["method"] for r in rows if r["family"]==fam)):
        sub=[r for r in rows if r["family"]==fam and r["method"]==meth]
        print("   %-40s %s"%(meth,L(sub)))

print()
print("########## PUBLIC SET (Sec VI-H) ##########")
for k,m in sorted(grp("public_dataset_sensitivity", lambda r:(r["public"],r["public_size"])).items()):
    print("   public=%s size=%s"%k)
    for meth in sorted(m): print("      %-40s %s"%(meth,L(m[meth])))

print()
print("########## CHANNEL SWEEP (Sec VI-G) ##########")
for k,m in sorted(grp("ablation_channel_sweep", lambda r:r["cfg"].get("downlink_snr_db")).items(),
                  key=lambda kv:(kv[0] is None, kv[0])):
    print("   DL SNR=%s"%k)
    for meth in sorted(m): print("      %-40s %s"%(meth,L(m[meth])))

print()
print("########## COEFFICIENT GRID (Sec VI-D) ##########")
cg=[r for r in rows if r["family"]=="coefficient_grid" and r["method"]==S]
accs=[r["acc"]*100 for r in cg]; gin=set(round(r["gini"]*100,4) for r in cg)
cells=collections.defaultdict(list)
for r in cg: cells[(r["cfg"].get("scope_alpha_uncertainty"),r["cfg"].get("scope_alpha_diversity"))].append(r["acc"]*100)
cm={k:sum(v)/len(v) for k,v in cells.items()}
print("   arms=%d cells=%d  acc range %.2f .. %.2f (spread %.2f)"%(len(cg),len(cells),min(accs),max(accs),max(accs)-min(accs)))
print("   cell-mean range %.2f .. %.2f (spread %.2f)"%(min(cm.values()),max(cm.values()),max(cm.values())-min(cm.values())))
print("   distinct gini values:",gin)
base=[k for k in cm if k==(0.3,0.1)]
if base: print("   cell (0.3,0.1) = %.2f"%cm[base[0]])
