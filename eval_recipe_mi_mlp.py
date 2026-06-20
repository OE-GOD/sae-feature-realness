import torch, numpy as np, json

torch.manual_seed(0); np.random.seed(0)

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
trF = d["train_F"].float().numpy(); teF = d["test_F"].float().numpy(); ooF = d["ood_F"].float().numpy()

SEL="mi"; NF=30; PROBE="mlp"

def f1(y, pred):
    y=y.astype(int); pred=pred.astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    if tp==0: return 0.0
    p=tp/(tp+fp); r=tp/(tp+fn)
    return 2*p*r/(p+r)

def select_mi(F, y, k):
    # binarize feature (>0), label 0/1, MI from 2x2 contingency
    n = len(y)
    yb = (y==1)
    fb = (F > 0)
    py1 = yb.mean(); py0 = 1-py1
    mis = np.zeros(F.shape[1])
    for j in range(F.shape[1]):
        f = fb[:, j]
        px1 = f.mean(); px0 = 1-px1
        mi = 0.0
        for fv, pxv in [(True, px1), (False, px0)]:
            for yv, pyv in [(True, py1), (False, py0)]:
                cnt = np.sum((f == fv) & (yb == yv))
                if cnt == 0: continue
                pxy = cnt / n
                if pxv<=0 or pyv<=0: continue
                mi += pxy * np.log(pxy / (pxv * pyv))
        mis[j] = mi
    return np.argsort(-mis)[:k]

def best_thresh(probs, y):
    bestf, bestt = -1, 0.5
    for t in np.unique(probs):
        f = f1(y, (probs>=t).astype(int))
        if f>bestf: bestf,bestt = f,t
    return bestt

class MLP(torch.nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(din,32), torch.nn.ReLU(), torch.nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def train_mlp(X, y):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    m = MLP(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    m.train()
    for _ in range(300):
        opt.zero_grad()
        out = m(Xt)
        loss = lossf(out, yt)
        loss.backward(); opt.step()
    return m

def proba(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X,dtype=torch.float32))).numpy()

rows=[]
for C in keys:
    ytr = d["train_L"][C].numpy().astype(int)
    if ytr.sum()<5: continue
    yte = d["test_L"][C].numpy().astype(int)
    yoo = d["ood_L"][C].numpy().astype(int)
    cols = select_mi(trF, ytr, NF)
    Xtr = trF[:,cols]; mu=Xtr.mean(0); sd=Xtr.std(0)+1e-8
    Xtr=(Xtr-mu)/sd; Xte=(teF[:,cols]-mu)/sd; Xoo=(ooF[:,cols]-mu)/sd
    m = train_mlp(Xtr, ytr)
    ptr=proba(m,Xtr)
    t=best_thresh(ptr,ytr)
    f_te=f1(yte,(proba(m,Xte)>=t).astype(int))
    f_oo=f1(yoo,(proba(m,Xoo)>=t).astype(int))
    rows.append((C,float(f_te),float(f_oo)))
    print(C,round(f_te,4),round(f_oo,4))

mte=float(np.mean([r[1] for r in rows])); moo=float(np.mean([r[2] for r in rows]))
print("MEAN_INDIST",mte); print("MEAN_OOD",moo)
print("JSON", json.dumps(rows))
