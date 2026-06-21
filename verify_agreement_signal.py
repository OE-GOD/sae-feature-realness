"""INDEPENDENT re-verification of the published claim (post 5): reliable-feature AGREEMENT
beats plain confidence AND a random-feature control at OOD abstention. Saved as a script
this time (the original was inline). If this doesn't reproduce, the post gets retracted."""
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
d=torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
more=torch.load("/Users/oe/rebuild/more_ood_sentiment.pt", weights_only=False)
OOD={"amazon":(d["am"][0],d["am"][1]),"yelp":(more["yelp"][0],more["yelp"][1]),"imdb":(more["imdb"][0],more["imdb"][1])}
ytr=d["train"][1].numpy(); yrt=d["rt"][1].numpy()
def scorr(Z,y): yc=y-y.mean();zc=Z-Z.mean(0);return np.nan_to_num((zc*yc[:,None]).mean(0)/(zc.std(0)*yc.std()+1e-9))
def selacc(pf,y,keep,base): return (pf[keep]==y[keep]).mean() if keep.sum()>0 else base
print("RE-VERIFY published claim: reliable-feature agreement vs confidence vs random control")
for pool in ["max","mean"]:
    Xtr=d["train"][0][pool].float().numpy(); mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; Ztr=(Xtr-mu)/sd
    Zrt=(d["rt"][0][pool].float().numpy()-mu)/sd
    c_tr=scorr(Ztr,ytr); c_rt=scorr(Zrt,yrt); freq=(Xtr>0).mean(0); alive=np.where(freq>0.01)[0]
    rel=alive[np.argsort(-(c_tr[alive]*c_rt[alive]))[:500]]
    cf=LogisticRegression(max_iter=500,C=0.3).fit(Ztr,ytr); cr=LogisticRegression(max_iter=500,C=0.3).fit(Ztr[:,rel],ytr)
    print(f"\n[{pool}] {'domain':7} {'base':>6} {'conf@cov':>9} {'RELIABLE':>9} {'RANDOM(5s)':>12}")
    for name,(P,yt) in OOD.items():
        Z=(P[pool].float().numpy()-mu)/sd; y=yt.numpy(); pf=cf.predict(Z); base=(pf==y).mean()
        ag=(pf==cr.predict(Z[:,rel])); cov=ag.mean(); acc_rel=selacc(pf,y,ag,base)
        conf=np.abs(cf.decision_function(Z)); thr=np.quantile(conf,1-cov); acc_conf=selacc(pf,y,conf>=thr,base)
        rnds=[selacc(pf,y,(pf==LogisticRegression(max_iter=500,C=0.3).fit(Ztr[:,(rr:=np.random.RandomState(s).choice(alive,500,replace=False))],ytr).predict(Z[:,rr])),base) for s in range(5)]
        print(f"     {name:7} {base:>6.3f} {acc_conf:>9.3f} {acc_rel:>9.3f} {np.mean(rnds):>9.3f}±{np.std(rnds):.2f}")
