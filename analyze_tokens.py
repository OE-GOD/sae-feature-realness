"""Decisive test: with the EXACT-ANSWER token (Orgad fix), is factual truth decodable, and does it
recover the model's OWN errors? Balance-robust AUROC on the error set (errors are ~91% one class).
Compare 'last' (last claim token) vs 'final' (answer-commit position), across layers, cross-topic."""
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
d=torch.load("factual_tokens.pt", weights_only=False)
LAYERS=[6,9,12,15,18]; INDIST=["cities","companies"]; OOD=["animals","inventions","elements"]
def auroc(s,yb):
    p=s[yb==1];n=s[yb==0]
    if len(p)==0 or len(n)==0: return float('nan')
    r=rankdata(np.concatenate([p,n])); return (r[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))
print(f"{'feat':12}{'truth_acc_OOD':>14}{'truth_AUROC_OOD':>16}{'err_AUROC':>11}{'correct_AUROC':>14}")
# confidence baseline for correctness
gap=np.concatenate([np.abs(d[t]["gap"]) for t in OOD]); cor=np.concatenate([(d[t]["judge"]==d[t]["truth"]).astype(int) for t in OOD])
print(f"{'(conf base)':12}{'':>14}{'':>16}{'':>11}{auroc(gap,cor):>14.3f}")
for L in LAYERS:
    for pos in ["last","final"]:
        k=f"L{L}_{pos}"
        Xin=np.vstack([d[t][k].float().numpy() for t in INDIST]); tin=np.concatenate([d[t]["truth"] for t in INDIST])
        cin=np.concatenate([(d[t]["judge"]==d[t]["truth"]).astype(int) for t in INDIST])
        mu,sd=Xin.mean(0),Xin.std(0)+1e-6
        tclf=LogisticRegression(max_iter=1000,C=0.5).fit((Xin-mu)/sd,tin)
        cclf=LogisticRegression(max_iter=1000,C=0.5).fit((Xin-mu)/sd,cin) if len(set(cin))>1 else None
        tacc=[];tsc=[];tt=[];esc=[];et=[];csc=[];ct=[]
        for t in OOD:
            Z=(d[t][k].float().numpy()-mu)/sd; truth=d[t]["truth"]; judge=d[t]["judge"]; err=(judge!=truth)
            s=tclf.predict_proba(Z)[:,1]; tacc.append((tclf.predict(Z)==truth).mean())
            tsc.append(s);tt.append(truth); esc.append(s[err]);et.append(truth[err])
            if cclf is not None: csc.append(cclf.predict_proba(Z)[:,1]);ct.append((judge==truth).astype(int))
        cau=auroc(np.concatenate(csc),np.concatenate(ct)) if csc else float('nan')
        print(f"{k:12}{np.mean(tacc):>14.3f}{auroc(np.concatenate(tsc),np.concatenate(tt)):>16.3f}"
              f"{auroc(np.concatenate(esc),np.concatenate(et)):>11.3f}{cau:>14.3f}")
print("\nerr_AUROC>0.5 = truth recoverable on the model's OWN errors (the key number). correct_AUROC vs conf base = metacognition.")
