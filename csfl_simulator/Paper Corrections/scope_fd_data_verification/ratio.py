from load import load, agg
import collections
rows=[r for r in load() if not r["incomplete"]]
S="fd_native.scope_fd"; D="fd_native.scope_fd_debt_only"; RND="heuristic.random"

def _rank(x):
    o=sorted(range(len(x)),key=lambda i:x[i]); rk=[0.0]*len(x); i=0
    while i<len(o):
        j=i
        while j+1<len(o) and x[o[j+1]]==x[o[i]]: j+=1
        avg=(i+j)/2.0+1
        for k in range(i,j+1): rk[o[k]]=avg
        i=j+1
    return rk
def spearman(x,y):
    rx,ry=_rank(x),_rank(y); n=len(x)
    mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    dx=sum((a-mx)**2 for a in rx)**.5; dy=sum((b-my)**2 for b in ry)**.5
    return num/(dx*dy)

# paired configs where scope, debt and random all present at alpha=0.5 FMNIST
cfgs=collections.defaultdict(lambda: collections.defaultdict(dict))
for r in rows:
    if r["dataset"]!="Fashion-MNIST" or r["alpha"]!=0.5: continue
    if r["family"] in ("dropout","channel_energy","public_dataset_sensitivity",
                       "histogram_privacy","bounded_staleness","coefficient_grid",
                       "ablation_channel_sweep"): continue
    cfgs[(r["N"],r["K"])][r["method"]][r["seed"]]=r["acc"]*100

out=[]
for (N,K),m in sorted(cfgs.items()):
    if RND not in m or S not in m: continue
    seeds=sorted(set(m[S])&set(m[RND]))
    if not seeds: continue
    gapS=sum(m[S][s]-m[RND][s] for s in seeds)/len(seeds)
    gapD=None
    if D in m:
        sd=sorted(set(m[D])&set(m[RND]))
        if sd: gapD=sum(m[D][s]-m[RND][s] for s in sd)/len(sd)
    out.append((K/float(N),N,K,gapS,gapD,len(seeds)))

print("%-8s %-5s %-4s %8s %8s %s"%("K/N","N","K","gap_scope","gap_debt","seeds"))
for ratio,N,K,gs,gd,n in sorted(out):
    print("%-8.4f %-5d %-4d %+8.2f %s   %d"%(ratio,N,K,gs,("%+8.2f"%gd) if gd is not None else "     n/a",n))
print()
print("configurations:",len(out))
xs=[o[0] for o in out]; ys=[o[3] for o in out]
print("Spearman(K/N, scope gap) = %.3f"%spearman(xs,ys))
pos=sum(1 for o in out if o[3]>0)
print("positive scope gaps: %d / %d"%(pos,len(out)))
dd=[o for o in out if o[4] is not None]
print("debt-only gaps: %d configs, range %+.2f .. %+.2f, positive %d/%d"
      %(len(dd),min(o[4] for o in dd),max(o[4] for o in dd),sum(1 for o in dd if o[4]>0),len(dd)))
print("scope gap range: %+.2f .. %+.2f"%(min(ys),max(ys)))
neg=[o for o in out if o[3]<0]
print("negative scope configs:",[(o[1],o[2]) for o in neg])
