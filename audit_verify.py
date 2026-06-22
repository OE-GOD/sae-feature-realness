"""Verify the diagnostic fleet's THREE consequential claims before believing them:
(1) the two poolings per domain are correlated -> effective n < 6, so domain-clustered
    significance is the honest test; (2) the AUGRC tie-break favors the binary champion via
    dataset row order; (3) the 'signed reliable attribution' rescue reproduces & is a
    disagreement-family signal."""
import numpy as np
from idea_harness import H, OOD, _auroc, _augrc
rng = np.random.RandomState(0)

def conf(h): return lambda Z: np.abs(h.full.decision_function(Z))
def champ(h):
    rel=h.reliable; rp=h.probe(cols=rel)
    return lambda Z:(h.full.predict(Z)==rp.predict(Z[:,rel])).astype(float)
def surp(h):
    rel=h.reliable; rp=h.probe(cols=rel)
    def s(Z):
        pr=np.clip(rp.predict_proba(Z[:,rel])[:,1],1e-6,1-1e-6); yh=h.full.predict(Z)
        return yh*np.log(pr)+(1-yh)*np.log(1-pr)
    return s
def attr_signed(h):  # the claimed rescue: signed reliable margin toward predicted class
    rel=h.reliable; rp=h.probe(cols=rel)
    def s(Z):
        m=rp.decision_function(Z[:,rel]); yh=h.full.predict(Z)
        return np.where(yh==1, m, -m)   # signed: high when reliable probe agrees w/ full's label
    return s

# collect per (domain,pool): trust arrays + correct (instances shared across poolings within domain)
data={}
for pool in ["mean","max"]:
    h=H(pool)
    fns={"conf":conf(h),"champ":champ(h),"surp":surp(h),"attr":attr_signed(h)}
    for dom,(P,yt) in OOD.items():
        Z=h.std(P[pool].float().numpy()); y=yt.numpy(); correct=(h.full.predict(Z)==y).astype(int)
        data[(dom,pool)]={k:np.asarray(f(Z),float) for k,f in fns.items()}; data[(dom,pool)]["correct"]=correct

doms=["amazon","yelp","imdb"]
print("(1) POOLING CORRELATION within domain (corr of correctness mean-vs-max):")
for d in doms:
    cm,cx=data[(d,"mean")]["correct"],data[(d,"max")]["correct"]
    print(f"    {d}: r={np.corrcoef(cm,cx)[0,1]:+.2f}")

def auroc_sig(sig): return {k:_auroc(data[k][sig],data[k]["correct"]) for k in data}
A={s:auroc_sig(s) for s in ["conf","champ","surp","attr"]}
print("\n(2) per-DOMAIN AUROC (avg of 2 poolings):")
print(f"    {'domain':8}{'conf':>7}{'champ':>7}{'surp':>7}{'attr':>7}")
for d in doms:
    print(f"    {d:8}"+"".join(f"{np.mean([A[s][(d,'mean')],A[s][(d,'max')]]):>7.3f}" for s in ["conf","champ","surp","attr"]))

# domain-clustered bootstrap: resample 3 domains w/ replacement; within each, resample instances
# (shared index across poolings -> respects pooling correlation). gap = mean AUROC diff over conditions.
def clustered_gap(sigA, sigB, B=3000):
    gaps=[]
    for _ in range(B):
        chosen=rng.choice(doms,3,replace=True); vals=[]
        for d in chosen:
            n=len(data[(d,'mean')]["correct"]); idx=rng.randint(0,n,n)
            for pool in ["mean","max"]:
                c=data[(d,pool)]["correct"][idx]
                vals.append(_auroc(data[(d,pool)][sigA][idx],c)-_auroc(data[(d,pool)][sigB][idx],c))
        gaps.append(np.nanmean(vals))
    gaps=np.array(gaps); return np.nanmean(gaps), np.nanmean(gaps>0)
print("\n(2b) DOMAIN-CLUSTERED bootstrap AUROC gaps (honest power):")
for a,b in [("surp","conf"),("champ","conf"),("surp","champ"),("attr","conf")]:
    g,p=clustered_gap(a,b); print(f"    {a:5} - {b:5}: mean gap {g:+.3f}, P(>0)={p:.3f}")

print("\n(3) AUGRC tie-break bias for the BINARY champion (row-order vs randomized):")
tot=[]
for k in data:
    t=data[k]["champ"]; c=data[k]["correct"]
    rowg=_augrc(t,c)
    rnd=np.mean([_augrc(t+rng.uniform(0,1e-6,len(t)),c) for _ in range(200)])
    tot.append(rnd-rowg)
print(f"    mean AUGRC inflation from row-order tie-break: {np.mean(tot):+.4f} (positive = champion was flattered)")

print("\n(4) does 'signed attribution' rescue == disagreement family? corr(attr, surp) per condition:")
print("    "+", ".join(f"{k[0][:3]}/{k[1][:3]}:{np.corrcoef(data[k]['attr'],data[k]['surp'])[0,1]:.2f}" for k in data))
