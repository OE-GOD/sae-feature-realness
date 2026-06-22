"""Resolve the recover_err puzzle: is mid-layer 'recovery' real latent knowledge or a class-balance
artifact? On the model's OWN errors (OOD pooled), use AUROC(probe_score, truth) which is balance-robust,
report class balance, and compare RAW RESIDUAL vs SAE features at layer 12 (does the SAE destroy facts?)."""
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
dl=torch.load("factual_layers.pt", weights_only=False)
dsae=torch.load("factual_latent.pt", weights_only=False)
INDIST=["cities","companies"]; OOD=["animals","inventions","elements"]
def auroc(s,yb):
    p=s[yb==1];n=s[yb==0]
    if len(p)==0 or len(n)==0: return float('nan')
    r=rankdata(np.concatenate([p,n])); return (r[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))

def eval_feats(get, name):
    Xin=np.vstack([get(t,"in") for t in INDIST]); tin=np.concatenate([dl[t]["truth"] for t in INDIST])
    mu,sd=Xin.mean(0),Xin.std(0)+1e-6
    clf=LogisticRegression(max_iter=1000,C=0.5).fit((Xin-mu)/sd,tin)
    es,et,allpred,alltruth=[],[],[],[]
    for t in OOD:
        Z=(get(t,"ood")-mu)/sd; truth=dl[t]["truth"]; judge=dl[t]["judge"]; err=(judge!=truth)
        sc=clf.predict_proba(Z)[:,1]; pred=clf.predict(Z)
        es.append(sc[err]); et.append(truth[err]); allpred.append(pred); alltruth.append(truth)
    es=np.concatenate(es);et=np.concatenate(et)
    acc_on_err=( (es>0.5).astype(int)==et ).mean()
    print(f"  {name:22} err_n={len(et)} true=1 frac={et.mean():.2f}  pred=1 frac={(es>0.5).mean():.2f}  "
          f"acc_on_err={acc_on_err:.2f}  AUROC_on_err={auroc(es,et):.3f}  overall_acc={np.mean([(np.concatenate(allpred)==np.concatenate(alltruth)).mean()]):.3f}")

print("LAYER-12 raw residual vs SAE features (does the SAE destroy factual info?):")
eval_feats(lambda t,_: dl[t]["L12"].float().numpy(), "raw resid L12")
eval_feats(lambda t,_: dsae[t]["max"].float().numpy(), "SAE L12 (max)")
eval_feats(lambda t,_: dsae[t]["mean"].float().numpy(), "SAE L12 (mean)")
print("\nAUROC_on_err > 0.5 means the probe genuinely ranks truth on the model's errors (balance-robust).")
print("acc_on_err can be inflated by class skew; trust AUROC_on_err.")
