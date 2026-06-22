"""Refined dig-out: (1) ortho-erase = erase the topic direction AFTER removing its sentiment
component (so we delete pure topic, not the answer); (2) mass-mean = difference-of-class-means
readout on reliable features (more shift-robust than logistic). Beat the working method
(reliable-core retrain, ~0.82)?  All leak-free (no OOD sentiment labels)."""
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from idea_harness import H, OOD as REV
hs=torch.load("hard_shift_sentiment.pt", weights_only=False); ALL=dict(REV); ALL.update(hs)

def unit(v):
    n=np.linalg.norm(v); return v/n if n>1e-9 else v

print(f"{'domain':10}{'pool':5}{'full':>7}{'reliable_core':>14}{'ortho_erase':>12}{'massmean_rel':>13}")
res={'reliable':[], 'ortho':[], 'mm':[]}
for pool in ["mean","max"]:
    h=H(pool); Ztr=h.Ztr; ytr=h.ytr; rel=h.reliable
    rp=h.probe(cols=rel)
    sent_dir=unit(h.scorr(Ztr,ytr))                      # sentiment direction (in-dist, leak-free)
    # mass-mean on reliable features: difference of class means
    mu1=Ztr[ytr==1][:,rel].mean(0); mu0=Ztr[ytr==0][:,rel].mean(0); mm=unit(mu1-mu0)
    mm_thresh=((Ztr[:,rel]@mm)).mean()
    for d,(P,y) in ALL.items():
        Z=h.std(P[pool].float().numpy()); yy=y.numpy()
        full=(h.full.predict(Z)==yy).mean(); relacc=(rp.predict(Z[:,rel])==yy).mean()
        # ortho-erase: topic dir = in-dist vs OOD, minus its sentiment component
        lab=np.r_[np.zeros(len(Ztr)),np.ones(len(Z))]
        wdom=LogisticRegression(max_iter=1000,C=1.0).fit(np.vstack([Ztr,Z]),lab).coef_[0]
        wdom=wdom-(wdom@sent_dir)*sent_dir; wdom=unit(wdom)   # remove sentiment component
        def proj(X): return X-(X@wdom)[:,None]*wdom
        clf=LogisticRegression(max_iter=1000,C=0.3).fit(proj(Ztr),ytr)
        ortho=(clf.predict(proj(Z))==yy).mean()
        mmacc=(((Z[:,rel]@mm)>mm_thresh).astype(int)==yy).mean()
        res['reliable'].append(relacc); res['ortho'].append(ortho); res['mm'].append(mmacc)
        print(f"{d:10}{pool:5}{full:>7.3f}{relacc:>14.3f}{ortho:>12.3f}{mmacc:>13.3f}")
print(f"\nMEAN  reliable-core {np.mean(res['reliable']):.3f}  ortho-erase {np.mean(res['ortho']):.3f}  massmean {np.mean(res['mm']):.3f}")
