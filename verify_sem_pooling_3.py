"""Fast targeted: P/R/F1 on rt+am for the two relevant recipes, 3 seeds, + apples-to-apples."""
import torch, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, precision_score, recall_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
def X(s,p): return data[s][0][p].float().numpy()
def Y(s): return data[s][1].numpy()

def select_cols(Xs, y, method, nf, seed):
    if method=="mi":
        return np.argsort(-mutual_info_classif(Xs,y,random_state=seed))[:nf]
    if method=="corr":
        yc=y-y.mean(); Xc=Xs-Xs.mean(0)
        num=(Xc*yc[:,None]).sum(0); den=np.sqrt((Xc**2).sum(0)*(yc**2).sum())+1e-9
        return np.argsort(-np.abs(num/den))[:nf]

def run(pool, method, pk, nf, seed):
    Xtr,ytr=X("train",pool),Y("train")
    mu,sd=Xtr.mean(0),Xtr.std(0); sd[sd==0]=1.0
    Z=lambda A:(A-mu)/sd; Xtr_s=Z(Xtr)
    cols=select_cols(Xtr_s,ytr,method,nf,seed)
    probe=LogisticRegression(max_iter=1000,random_state=seed) if pk=="linear" else \
          MLPClassifier(hidden_layer_sizes=(32,),alpha=1e-3,max_iter=400,random_state=seed)
    probe.fit(Xtr_s[:,cols],ytr)
    o={}
    for sp in ["test","rt","am"]:
        yp=probe.predict(Z(X(sp,pool))[:,cols]); yt=Y(sp)
        o[sp]=dict(f1=f1_score(yt,yp,zero_division=0),p=precision_score(yt,yp,zero_division=0),
                   r=recall_score(yt,yp,zero_division=0),pos=float(yp.mean()))
    o["mood"]=(o["rt"]["f1"]+o["am"]["f1"])/2
    return o

def threeseed(pool,m,pk,nf):
    rs=[run(pool,m,pk,nf,s) for s in [0,1,2]]
    return rs

def report(label,pool,m,pk,nf):
    rs=threeseed(pool,m,pk,nf)
    mood=np.array([r["mood"] for r in rs])
    print(f"\n[{label}] pool={pool} recipe={m}/{pk}/nf={nf}")
    print(f"  mean_OOD_F1 seeds={np.round(mood,3).tolist()} mean={mood.mean():.3f} std={mood.std():.3f} range={mood.max()-mood.min():.3f}")
    for sp in ["test","rt","am"]:
        f=np.array([r[sp]["f1"] for r in rs]); p=np.array([r[sp]["p"] for r in rs])
        rr=np.array([r[sp]["r"] for r in rs]); pos=np.array([r[sp]["pos"] for r in rs])
        print(f"  {sp:4s}: F1={f.mean():.3f}+/-{f.std():.3f}  P={p.mean():.3f}  R={rr.mean():.3f}  pred_pos_rate={pos.mean():.3f} (base 0.5)")
    return mood

print("=== TWO BEST RECIPES, 3 seeds, full P/R/F1 ===")
mean_mood = report("BEST-MEAN","mean","mi","linear",100)
max_mood  = report("BEST-MAX","max","corr","linear",100)

print("\n=== HEADLINE GAP: best-max vs best-mean (each its own best recipe) ===")
g=max_mood-mean_mood
print(f"max {max_mood.mean():.3f}  vs  mean {mean_mood.mean():.3f}  gap={g.mean():+.3f}+/-{g.std():.3f}  per-seed={np.round(g,3).tolist()}")

print("\n=== APPLES-TO-APPLES: same recipe, swap only pooling ===")
# mean's recipe on max
print("-- recipe mi/linear/100 (mean's best) --")
a=np.array([r["mood"] for r in threeseed("mean","mi","linear",100)])
b=np.array([r["mood"] for r in threeseed("max","mi","linear",100)])
print(f"  mean={a.mean():.3f} max={b.mean():.3f} gap(max-mean)={ (b-a).mean():+.3f}+/-{(b-a).std():.3f} per-seed={np.round(b-a,3).tolist()}")
# max's recipe on mean
print("-- recipe corr/linear/100 (max's best) --")
c=np.array([r["mood"] for r in threeseed("max","corr","linear",100)])
d=np.array([r["mood"] for r in threeseed("mean","corr","linear",100)])
print(f"  max={c.mean():.3f} mean={d.mean():.3f} gap(max-mean)={ (c-d).mean():+.3f}+/-{(c-d).std():.3f} per-seed={np.round(c-d,3).tolist()}")
