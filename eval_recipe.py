import torch, numpy as np, json

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
trF = d["train_F"].float().numpy(); teF = d["test_F"].float().numpy(); ooF = d["ood_F"].float().numpy()

SEL="corr"; NF=30

def f1(y, pred):
    y=y.astype(int); pred=pred.astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    if tp==0: return 0.0
    p=tp/(tp+fp); r=tp/(tp+fn)
    return 2*p*r/(p+r)

def select_corr(F, y, k):
    yc = y - y.mean()
    Fc = F - F.mean(0)
    num = (Fc * yc[:,None]).sum(0)
    den = np.sqrt((Fc**2).sum(0) * (yc**2).sum() + 1e-12)
    r = np.abs(num/den)
    r = np.nan_to_num(r)
    return np.argsort(-r)[:k]

def train_logreg(X, y, lr=0.1, iters=2000, l2=1e-4):
    n,m = X.shape
    w = np.zeros(m); b = 0.0
    yf = y.astype(float)
    for _ in range(iters):
        z = X@w + b
        p = 1/(1+np.exp(-z))
        g = p - yf
        gw = X.T@g/n + l2*w
        gb = g.mean()
        w -= lr*gw; b -= lr*gb
    return w,b

def proba(X,w,b):
    return 1/(1+np.exp(-(X@w+b)))

def best_thresh(probs, y):
    ts = np.unique(probs)
    bestf, bestt = -1, 0.5
    for t in ts:
        f = f1(y, (probs>=t).astype(int))
        if f>bestf: bestf,bestt = f,t
    return bestt

rows=[]
for C in keys:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum()<5: continue
    yte = d["test_L"][C].numpy().astype(int)
    yoo = d["ood_L"][C].numpy().astype(int)
    cols = select_corr(trF, ytr.astype(float), NF)
    Xtr = trF[:,cols]; mu=Xtr.mean(0); sd=Xtr.std(0)+1e-8
    Xtr=(Xtr-mu)/sd; Xte=(teF[:,cols]-mu)/sd; Xoo=(ooF[:,cols]-mu)/sd
    w,b = train_logreg(Xtr, ytr)
    ptr=proba(Xtr,w,b)
    t=best_thresh(ptr,ytr)
    f_te=f1(yte,(proba(Xte,w,b)>=t).astype(int))
    f_oo=f1(yoo,(proba(Xoo,w,b)>=t).astype(int))
    rows.append((C,float(f_te),float(f_oo)))
    print(C,f_te,f_oo)

mte=np.mean([r[1] for r in rows]); moo=np.mean([r[2] for r in rows])
print("MEAN_INDIST",mte); print("MEAN_OOD",moo)
print(json.dumps(rows))
