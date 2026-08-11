from load import load, agg
import collections
rows=[r for r in load() if not r["incomplete"]]
S="fd_native.scope_fd"; D="fd_native.scope_fd_debt_only"; RND="heuristic.random"
def L(v):
    a,sd,n=agg([x["acc"]*100 for x in v]); g,_,_=agg([x["gini"]*100 for x in v])
    return "acc=%6.2f+-%4.2f gini=%6.2f n=%d"%(a,sd,g,n)
def grp(fam,key,filt=None):
    o=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["family"]==fam and (filt is None or filt(r)): o[key(r)][r["method"]].append(r)
    return o

print("########## AUDIO FSDD (Sec VI-M) -- paper uses local_epochs=3, public 150, 5 seeds")
for k,m in sorted(grp("audio_fsdd", lambda r:(r["local_epochs"],r["public_size"])).items()):
    print("  local_epochs=%s public=%s"%k)
    for meth in sorted(m): print("     %-40s %s"%(meth,L(m[meth])))

print()
print("########## DROPOUT by p_drop (Sec VI-J)")
for k,m in sorted(grp("dropout", lambda r:r["cfg"]["dropout_prob"]).items()):
    print("  p_drop=%s"%k)
    for meth in [S,RND,D]:
        if m.get(meth): print("     %-40s %s"%(meth,L(m[meth])))

print()
print("########## STALENESS by W (Sec VI-J)")
for k,m in sorted(grp("bounded_staleness", lambda r:r["cfg"]["staleness_window"]).items()):
    print("  W=%s"%k)
    for meth in [S,RND]:
        if m.get(meth): print("     %-40s %s"%(meth,L(m[meth])))

print()
print("########## CHANNEL SWEEP by DL SNR (Sec VI-G)")
for k,m in sorted(grp("ablation_channel_sweep", lambda r:r["cfg"]["dl_snr_db"]).items()):
    print("  DL SNR=%s dB"%k)
    for meth in sorted(m): print("     %-40s %s"%(meth,L(m[meth])))

print()
print("########## CHANNEL/ENERGY (Sec VI-K) by DL SNR")
for k,m in sorted(grp("channel_energy", lambda r:r["cfg"].get("dl_snr_db")).items(),key=lambda kv:(kv[0] is None,kv[0])):
    print("  DL SNR=%s dB"%k)
    for meth in sorted(m): print("     %-40s %s"%(meth,L(m[meth])))

print()
print("########## PUBLIC SET incl. label noise (Sec VI-H)")
for k,m in sorted(grp("public_dataset_sensitivity",
        lambda r:(r["cfg"]["public_dataset"],r["cfg"]["public_dataset_size"],r["cfg"]["public_label_noise"])).items()):
    print("  public=%s size=%s noise=%s"%k)
    for meth in [S,D,RND]:
        if m.get(meth): print("     %-40s %s"%(meth,L(m[meth])))

print()
print("########## COEFFICIENT GRID (Sec VI-D)")
cells=collections.defaultdict(list)
for r in rows:
    if r["family"]=="coefficient_grid" and r["method"]==S:
        cells[(r["cfg"]["scope_au"],r["cfg"]["scope_ad"])].append(r["acc"]*100)
cm={k:sum(v)/len(v) for k,v in cells.items()}
lo=min(cm,key=cm.get); hi=max(cm,key=cm.get)
print("  cells=%d, seeds/cell=%d"%(len(cells),len(next(iter(cells.values())))))
print("  cell-mean min %.2f at au=%s ad=%s"%(cm[lo],lo[0],lo[1]))
print("  cell-mean max %.2f at au=%s ad=%s"%(cm[hi],hi[0],hi[1]))
print("  SPREAD = %.2f pp"%(cm[hi]-cm[lo]))
print("  cell (0.3,0.1) = %.2f"%cm[(0.3,0.1)])
g=set(round(r["gini"]*100,4) for r in rows if r["family"]=="coefficient_grid" and r["method"]==S)
print("  distinct gini across grid:",g)
