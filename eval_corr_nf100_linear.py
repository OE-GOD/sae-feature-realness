import torch, numpy as np, json

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
trF = d["train_F"].numpy().astype(np.float64)
teF = d["test_F"].numpy().astype(np.float64)
ooF = d["ood_F"].numpy().astype(np.float64)

NF=100

def select_corr(F, y, k):
    yc = y - y.mean()
    Fc = F - F.mean(0)
    num = (Fc * yc[:,None]).sum(0)
    den = np.sqrt((Fc**2).sum(0) * (yc**2).sum() + 1e-12)
    r = np.abs(num/den)
    return np.argsort(-r)[:k]

def sigmoid(z): return 1.0/(1.0+np.exp(-np.clip(z,-30,30)))

def train_logreg(X, y, iters=800, lr=0.5, l2=1e-4):
    n,dn = X.shape
    Xb = np.hstack([X, np.ones((n,1))])
    w = np.zeros(dn+1)
    for _ in range(iters):
        p = sigmoid(Xb@w)
        g = Xb.T@(p-y)/n
        g[:-1]+=l2*w[:-1]
        w -= lr*g
    return w

def predict(X,w):
    Xb=np.hstack([X,np.ones((X.shape[0],1))])
    return sigmoid(Xb@w)

def f1(y,pred):
    tp=np.sum((pred==1)&(y==1)); fp=np.sum((pred==1)&(y==0)); fn=np.sum((pred==0)&(y==1))
    if tp==0: return 0.0
    prec=tp/(tp+fp); rec=tp/(tp+fn)
    return 2*prec*rec/(prec+rec)

def best_thresh(probs,y):
    ts=np.unique(np.round(probs,4))
    bf,bt=-1,0.5
    for t in ts:
        f=f1(y,(probs>=t).astype(int))
        if f>bf: bf,bt=f,t
    return bt

rows=[]
for C in keys:
    ytr=d["train_L"][C].numpy().astype(np.float64)
    if ytr.sum()<5: continue
    yte=d["test_L"][C].numpy().astype(int)
    yoo=d["ood_L"][C].numpy().astype(int)
    cols=select_corr(trF,ytr,NF)
    Xtr=trF[:,cols]; mu=Xtr.mean(0); sd=Xtr.std(0)+1e-8
    Xtr=(Xtr-mu)/sd; Xte=(teF[:,cols]-mu)/sd; Xoo=(ooF[:,cols]-mu)/sd
    w=train_logreg(Xtr,ytr)
    ptr=predict(Xtr,w)
    t=best_thresh(ptr,ytr.astype(int))
    f_te=f1(yte,(predict(Xte,w)>=t).astype(int))
    f_oo=f1(yoo,(predict(Xoo,w)>=t).astype(int))
    rows.append((C,float(f_te),float(f_oo)))
    print(C,round(f_te,4),round(f_oo,4))

mte=float(np.mean([r[1] for r in rows])); moo=float(np.mean([r[2] for r in rows]))
print("MEAN_INDIST",round(mte,4)); print("MEAN_OOD",round(moo,4))
print(json.dumps(rows))
