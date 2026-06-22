"""Verify the fleet's headline: 'reliable-aligned evidence' (single-probe attribution) beats
surprisal on the hard shift. trust = sign(full margin) * (reliable features' contribution to the
full probe's own margin). Check: reproduces? leak-free? genuinely distinct from surprisal, or a
disguised member of the same reliable-feature-contrast family? Test on BOTH review and hard shift."""
import torch, numpy as np
from idea_harness import H, OOD, _auroc, _augrc
hs = torch.load("hard_shift_sentiment.pt", weights_only=False)
ALL = dict(OOD); ALL.update(hs)

def surp(h):
    rel=h.reliable; rp=h.probe(cols=rel)
    def s(Z):
        pr=np.clip(rp.predict_proba(Z[:,rel])[:,1],1e-6,1-1e-6); yh=h.full.predict(Z); return yh*np.log(pr)+(1-yh)*np.log(1-pr)
    return s
def evidence(h):  # reliable-aligned evidence: sign(full margin) * reliable contribution to full margin
    rel=h.reliable; w=h.full.coef_[0]
    def s(Z):
        m=h.full.decision_function(Z); contrib_rel=Z[:,rel]@w[rel]; return np.sign(m)*contrib_rel
    return s

print(f"{'domain':10}{'pool':5}{'base':>6}{'surp_AUC':>9}{'evid_AUC':>9}{'Δ':>7}{'corr':>7}")
rows=[]
for pool in ["mean","max"]:
    h=H(pool); sf=surp(h); ef=evidence(h)
    for d,(P,y) in ALL.items():
        Z=h.std(P[pool].float().numpy()); yy=y.numpy(); correct=(h.full.predict(Z)==yy).astype(int)
        sv=sf(Z); ev=ef(Z); sA=_auroc(sv,correct); eA=_auroc(ev,correct)
        cc=np.corrcoef(sv,ev)[0,1]; hard = d in hs
        rows.append((d,hard,correct.mean(),sA,eA,cc))
        print(f"{d:10}{pool:5}{correct.mean():>6.3f}{sA:>9.3f}{eA:>9.3f}{eA-sA:>+7.3f}{cc:>7.2f}")
r=rows
hard=[x for x in r if x[1]]; rev=[x for x in r if not x[1]]
print(f"\nHARD shift (tweet/poem/financial): evid beats surp in {sum(x[4]>x[3] for x in hard)}/{len(hard)}; "
      f"mean surp {np.mean([x[3] for x in hard]):.3f} vs evid {np.mean([x[4] for x in hard]):.3f}")
print(f"REVIEW OOD (amazon/yelp/imdb):     evid beats surp in {sum(x[4]>x[3] for x in rev)}/{len(rev)}; "
      f"mean surp {np.mean([x[3] for x in rev]):.3f} vs evid {np.mean([x[4] for x in rev]):.3f}")
print(f"mean corr(surp, evid) = {np.mean([x[5] for x in r]):.2f}  (high => same family, not orthogonal)")

# --- domain-clustered significance (n=3 hard domains is small; cluster by domain) ---
import numpy as np
rng=np.random.RandomState(0)
def sig(h):
    rel=h.reliable; rp=h.probe(cols=rel); w=h.full.coef_[0]
    def surp_s(Z):
        pr=np.clip(rp.predict_proba(Z[:,rel])[:,1],1e-6,1-1e-6); yh=h.full.predict(Z); return yh*np.log(pr)+(1-yh)*np.log(1-pr)
    def evid_s(Z):
        m=h.full.decision_function(Z); return np.sign(m)*(Z[:,rel]@w[rel])
    def conf_s(Z): return np.abs(h.full.decision_function(Z))
    return surp_s, evid_s, conf_s
hard_doms=["tweet","poem","financial"]
D={}
for pool in ["mean","max"]:
    h=H(pool); ss,es,cs=sig(h)
    for d in hard_doms:
        P,y=hs[d]; Z=h.std(P[pool].float().numpy()); c=(h.full.predict(Z)==y.numpy()).astype(int)
        D[(d,pool)]={"surp":ss(Z),"evid":es(Z),"conf":cs(Z),"c":c}
def clustered(aS,bS,B=4000):
    g=[]
    for _ in range(B):
        ch=rng.choice(hard_doms,3,replace=True); v=[]
        for d in ch:
            n=len(D[(d,'mean')]["c"]); idx=rng.randint(0,n,n)
            for pool in ["mean","max"]:
                c=D[(d,pool)]["c"][idx]; v.append(_auroc(D[(d,pool)][aS][idx],c)-_auroc(D[(d,pool)][bS][idx],c))
        g.append(np.nanmean(v))
    g=np.array(g); return np.nanmean(g), np.nanmean(g>0)
print("\n--- domain-clustered bootstrap (hard shift), honest power ---")
for a,b in [("evid","surp"),("evid","conf"),("surp","conf")]:
    m,p=clustered(a,b); print(f"  {a} - {b}: mean AUROC gap {m:+.3f}, P(>0)={p:.3f}")
