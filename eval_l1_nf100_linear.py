import torch, numpy as np, json

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
trF = d["train_F"].float().numpy(); teF = d["test_F"].float().numpy(); ooF = d["ood_F"].float().numpy()

NF = 100

def f1(y, pred):
    y=y.astype(int); pred=pred.astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    if tp==0: return 0.0
    p=tp/(tp+fp); r=tp/(tp+fn)
    return 2*p*r/(p+r)

def train_logreg(X, y, lr=0.1, iters=2000, l2=1e-4, l1=0.0):
    n,m = X.shape
    w = np.zeros(m); b = 0.0
    yf = y.astype(float)
    for _ in range(iters):
        z = X@w + b
        p = 1/(1+np.exp(-np.clip(z,-30,30)))
        g = p - yf
        gw = X.T@g/n + l2*w + l1*np.sign(w)
        gb = g.mean()
        w -= lr*gw; b -= lr*gb
    return w,b

def proba(X,w,b):
    return 1/(1+np.exp(-np.clip(X@w+b,-30,30)))

def best_thresh(probs, y):
    ts = np.unique(probs)
    bestf, bestt = -1, 0.5
    for t in ts:
        f = f1(y, (probs>=t).astype(int))
        if f>bestf: bestf,bestt = f,t
    return bestt

# Standardize ALL features once (train mean/std) for L1 selection
mu_all = trF.mean(0); sd_all = trF.std(0)+1e-8
trF_std_all = (trF - mu_all)/sd_all

rows=[]
for C in keys:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum()<5: continue
    yte = d["test_L"][C].numpy().astype(int)
    yoo = d["ood_L"][C].numpy().astype(int)

    # SELECTION: L1-penalized logreg on ALL 2048 standardized features, top-k by |weight|
    w_l1, _ = train_logreg(trF_std_all, ytr, lr=0.5, iters=400, l2=0.0, l1=1e-3)
    cols = np.argsort(-np.abs(w_l1))[:NF]

    # Standardize selected feats with train mean/std (raw feats)
    Xtr = trF[:,cols]; mu=Xtr.mean(0); sd=Xtr.std(0)+1e-8
    Xtr=(Xtr-mu)/sd; Xte=(teF[:,cols]-mu)/sd; Xoo=(ooF[:,cols]-mu)/sd

    # PROBE: linear logreg
    w,b = train_logreg(Xtr, ytr, lr=0.1, iters=2000, l2=1e-4)
    ptr=proba(Xtr,w,b)
    t=best_thresh(ptr,ytr)
    f_te=f1(yte,(proba(Xte,w,b)>=t).astype(int))
    f_oo=f1(yoo,(proba(Xoo,w,b)>=t).astype(int))
    rows.append((C,float(f_te),float(f_oo)))
    print(C,round(f_te,4),round(f_oo,4))

mte=float(np.mean([r[1] for r in rows])); moo=float(np.mean([r[2] for r in rows]))
print("MEAN_INDIST",round(mte,4)); print("MEAN_OOD",round(moo,4))
print(json.dumps({"mean_indist_f1":mte,"mean_ood_f1":moo,
    "per_concept":[{"concept":c,"indist_f1":a,"ood_f1":o} for c,a,o in rows]}))
