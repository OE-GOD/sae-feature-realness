"""Verify the headline: difference-of-means on reliable features recovers ~0.93 on near-chance
financial. Check class balance (no degenerate predictor), threshold sensitivity, bootstrap CI."""
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from idea_harness import H
hs=torch.load("hard_shift_sentiment.pt", weights_only=False)
def unit(v): n=np.linalg.norm(v); return v/n if n>1e-9 else v
rng=np.random.RandomState(0)
for pool in ["max","mean"]:
    h=H(pool); Ztr=h.Ztr; ytr=h.ytr; rel=h.reliable
    mu1=Ztr[ytr==1][:,rel].mean(0); mu0=Ztr[ytr==0][:,rel].mean(0); mm=unit(mu1-mu0)
    proj_tr=Ztr[:,rel]@mm
    thr_indist=proj_tr.mean(); thr_mid=(proj_tr[ytr==1].mean()+proj_tr[ytr==0].mean())/2
    P,y=hs["financial"]; Z=h.std(P[pool].float().numpy()); yy=y.numpy(); s=Z[:,rel]@mm
    for tname,thr in [("indist_mean",thr_indist),("class_mid",thr_mid),("zero",0.0)]:
        pred=(s>thr).astype(int); acc=(pred==yy).mean()
        print(f"[{pool}] financial  thr={tname:11} acc={acc:.3f}  pred_pos={pred.mean():.2f} (true_pos=0.50)")
    # bootstrap CI at class_mid threshold, paired vs reliable-core logistic
    rp=h.probe(cols=rel); rel_pred=rp.predict(Z[:,rel]); mm_pred=(s>thr_mid).astype(int)
    accs_mm=[];accs_rl=[]
    for _ in range(3000):
        bi=rng.randint(0,len(yy),len(yy)); accs_mm.append((mm_pred[bi]==yy[bi]).mean()); accs_rl.append((rel_pred[bi]==yy[bi]).mean())
    accs_mm=np.array(accs_mm);accs_rl=np.array(accs_rl)
    print(f"     mass-mean acc {accs_mm.mean():.3f} CI[{np.percentile(accs_mm,2.5):.3f},{np.percentile(accs_mm,97.5):.3f}]  "
          f"reliable-logistic {accs_rl.mean():.3f}  P(mm>logistic)={np.mean(accs_mm>accs_rl):.3f}")
