"""Frontier hypothesis: does the trust signal get MORE reliable as the detector gets LESS
reliable? Across ALL 6 OOD domains (amazon/yelp/imdb review + tweet/poem/financial hard),
correlate base detector accuracy with the trust signal's AUROC (correct-vs-wrong)."""
import torch, numpy as np
from idea_harness import H, OOD, _auroc
hs = torch.load("hard_shift_sentiment.pt", weights_only=False)
ALL = dict(OOD); ALL.update(hs)   # 6 domains

def surp(h):
    rel=h.reliable; rp=h.probe(cols=rel)
    def s(Z):
        pr=np.clip(rp.predict_proba(Z[:,rel])[:,1],1e-6,1-1e-6); yh=h.full.predict(Z); return yh*np.log(pr)+(1-yh)*np.log(1-pr)
    return s
def champ(h):
    rel=h.reliable; rp=h.probe(cols=rel); return lambda Z:(h.full.predict(Z)==rp.predict(Z[:,rel])).astype(float)

rows=[]
print(f"{'domain':10}{'pool':5}{'base_acc':>9}{'#wrong':>7}{'conf_AUC':>9}{'champ_AUC':>10}{'surp_AUC':>9}")
for pool in ["mean","max"]:
    h=H(pool); sc=surp(h); cm=champ(h)
    for d,(P,y) in ALL.items():
        Z=h.std(P[pool].float().numpy()); yy=y.numpy(); pf=h.full.predict(Z); correct=(pf==yy).astype(int)
        base=correct.mean(); nwrong=int((correct==0).sum())
        cA=_auroc(np.abs(h.full.decision_function(Z)),correct); chA=_auroc(cm(Z),correct); sA=_auroc(sc(Z),correct)
        rows.append((d,pool,base,nwrong,cA,chA,sA))
        print(f"{d:10}{pool:5}{base:>9.3f}{nwrong:>7}{cA:>9.3f}{chA:>10.3f}{sA:>9.3f}")

r=np.array([(x[2],x[4],x[5],x[6]) for x in rows])  # base, conf, champ, surp
base=r[:,0]
print(f"\n--- correlation across {len(rows)} (domain x pool) points: base accuracy vs trust AUROC ---")
for name,col in [("confidence",1),("binary agreement",2),("surprisal",3)]:
    cc=np.corrcoef(base, r[:,col])[0,1]
    print(f"  corr(base_acc, {name:18} AUROC) = {cc:+.3f}")
print(f"\n  mean trust AUROC when base<0.65 (detector failing): surprisal "
      f"{r[base<0.65,3].mean():.3f}, agreement {r[base<0.65,2].mean():.3f}, confidence {r[base<0.65,1].mean():.3f}")
print(f"  mean trust AUROC when base>=0.75 (detector okay):    surprisal "
      f"{r[base>=0.75,3].mean():.3f}, agreement {r[base>=0.75,2].mean():.3f}, confidence {r[base>=0.75,1].mean():.3f}")
