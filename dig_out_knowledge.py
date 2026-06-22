"""Frontier attempt to DIG OUT the latent answer: the correct sentiment is entangled with TOPIC.
So erase the topic/domain subspace (INLP, using only in-dist-vs-OOD identity — NO sentiment
labels on OOD = leak-free), then read sentiment. Does erasing topic recover the suppressed answer?"""
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from idea_harness import H
hs=torch.load("hard_shift_sentiment.pt", weights_only=False)
from idea_harness import OOD as REV; ALL=dict(REV); ALL.update(hs)

def inlp_dirs(Xin, Xood, k):
    """k orthonormal directions that distinguish in-dist from this OOD domain (topic directions)."""
    Xi, Xo = Xin.copy(), Xood.copy(); W=[]
    for _ in range(k):
        X=np.vstack([Xi,Xo]); lab=np.r_[np.zeros(len(Xi)), np.ones(len(Xo))]
        w=LogisticRegression(max_iter=200,C=1.0).fit(X,lab).coef_[0].astype(np.float64)
        for u in W: w-=(w@u)*u
        n=np.linalg.norm(w)
        if n<1e-8: break
        w/=n; W.append(w)
        Xi-=(Xi@w)[:,None]*w; Xo-=(Xo@w)[:,None]*w
    return np.array(W)

print(f"{'domain':10}{'pool':5}{'baseline':>9}{'erase_topic_k=5':>16}{'reliable_core':>14}")
for pool in ["mean","max"]:
    h=H(pool); Ztr=h.Ztr; ytr=h.ytr; rel=h.reliable; rp=h.probe(cols=rel)
    for d,(P,y) in ALL.items():
        Zood=h.std(P[pool].float().numpy()); yy=y.numpy()
        base=(h.full.predict(Zood)==yy).mean(); relacc=(rp.predict(Zood[:,rel])==yy).mean()
        # erase topic: directions from in-dist(train) vs this unlabeled OOD batch
        W=inlp_dirs(Ztr, Zood, 5)
        def proj(X): return X-(X@W.T)@W
        clf=LogisticRegression(max_iter=500,C=0.3).fit(proj(Ztr), ytr)   # retrain on topic-erased in-dist
        erased=(clf.predict(proj(Zood))==yy).mean()
        print(f"{d:10}{pool:5}{base:>9.3f}{erased:>16.3f}{relacc:>14.3f}")
