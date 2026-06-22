"""Two frontier questions on how the model represents facts, across layers (raw residual stream):
(A) WHERE is factual truth linearly decodable? cross-topic truth-probe accuracy per layer, and
    does it RECOVER the model's own errors (the suppressed truth) at any layer?
(B) METACOGNITION: does any layer's internal state predict the model's CORRECTNESS better than its
    own confidence (|True-False logit gap|)? i.e., does it sense its errors even if it can't fix them?"""
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
d=torch.load("factual_layers.pt", weights_only=False)
LAYERS=[3,6,9,12,15,18,21,24]; INDIST=["cities","companies"]; OOD=["animals","inventions","elements"]
def auroc(s,yb):
    p=s[yb==1];n=s[yb==0]
    if len(p)==0 or len(n)==0: return float('nan')
    r=rankdata(np.concatenate([p,n])); return (r[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))

# confidence baseline for (B): does |gap| predict correctness OOD?
gap_ood=np.concatenate([np.abs(d[t]["gap"]) for t in OOD])
corr_ood=np.concatenate([(d[t]["judge"]==d[t]["truth"]).astype(int) for t in OOD])
print(f"(B baseline) model confidence |logit gap| predicting its own correctness OOD: AUROC={auroc(gap_ood,corr_ood):.3f}")

print(f"\n{'layer':6}{'truth_acc_OOD':>14}{'recover_err':>12}{'correct_AUROC_internal':>24}")
for L in LAYERS:
    k=f"L{L}"
    Xin=np.vstack([d[t][k].float().numpy() for t in INDIST]);
    tin=np.concatenate([d[t]["truth"] for t in INDIST]); cin=np.concatenate([(d[t]["judge"]==d[t]["truth"]).astype(int) for t in INDIST])
    mu,sd=Xin.mean(0),Xin.std(0)+1e-6; Zin=(Xin-mu)/sd
    truth_clf=LogisticRegression(max_iter=1000,C=0.5).fit(Zin,tin)
    # correctness probe (metacognition): predict (judge==truth) from internal state
    corr_clf=LogisticRegression(max_iter=1000,C=0.5).fit(Zin,cin) if len(set(cin))>1 else None
    taccs=[];recs=[];cscore=[];ctrue=[]
    for t in OOD:
        Z=(d[t][k].float().numpy()-mu)/sd; truth=d[t]["truth"]; judge=d[t]["judge"]
        tp=truth_clf.predict(Z); taccs.append((tp==truth).mean())
        err=(judge!=truth); recs.append((tp[err]==truth[err]).mean() if err.sum() else np.nan)
        if corr_clf is not None:
            cscore.append(corr_clf.predict_proba(Z)[:,1]); ctrue.append((judge==truth).astype(int))
    cau=auroc(np.concatenate(cscore),np.concatenate(ctrue)) if cscore else float('nan')
    print(f"{L:6}{np.mean(taccs):>14.3f}{np.nanmean(recs):>12.1%}{cau:>24.3f}")
print("\n(A) truth_acc_OOD: where is factual truth decodable across layers; recover_err: does it recover the model's OWN errors")
print("(B) correct_AUROC_internal: internal metacognition vs the confidence baseline above")
