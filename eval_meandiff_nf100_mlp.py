import torch, numpy as np
torch.manual_seed(0); np.random.seed(0)

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
concepts = d["keys"]
trF = d["train_F"].float(); teF = d["test_F"].float(); ooF = d["ood_F"].float()

def f1_at(scores, y, thr):
    pred = (scores >= thr).astype(np.float32)
    tp = ((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()
    if tp==0: return 0.0
    p=tp/(tp+fp); r=tp/(tp+fn)
    return 2*p*r/(p+r) if (p+r)>0 else 0.0

def best_thr(scores, y):
    cand = np.unique(scores)
    if len(cand)>400:
        cand = np.quantile(scores, np.linspace(0,1,400))
    best=(0.0,cand[0])
    for t in cand:
        f=f1_at(scores,y,t)
        if f>best[0]: best=(f,t)
    return best[1]

class MLP(torch.nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(din,32), torch.nn.ReLU(), torch.nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

NF=100
rows=[]
for C in concepts:
    ytr = d["train_L"][C].float()
    if ytr.sum() < 5:
        continue
    yte = d["test_L"][C].float(); yoo = d["ood_L"][C].float()
    ytr_np=ytr.numpy(); yte_np=yte.numpy(); yoo_np=yoo.numpy()

    # meandiff selection on train
    m1 = trF[ytr==1].mean(0); m0 = trF[ytr==0].mean(0)
    md = (m1-m0).abs()
    cols = torch.topk(md, NF).indices

    Xtr = trF[:,cols]; Xte = teF[:,cols]; Xoo = ooF[:,cols]
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-8
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd; Xoo=(Xoo-mu)/sd

    torch.manual_seed(0)
    model=MLP(NF)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos=ytr.sum(); neg=len(ytr)-pos
    pw=torch.tensor(neg/max(pos,1.0))
    lossf=torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(300):
        opt.zero_grad()
        out=model(Xtr)
        loss=lossf(out, ytr)
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        str_=torch.sigmoid(model(Xtr)).numpy()
        ste=torch.sigmoid(model(Xte)).numpy()
        soo=torch.sigmoid(model(Xoo)).numpy()
    thr=best_thr(str_, ytr_np)
    f_in=f1_at(ste, yte_np, thr)
    f_oo=f1_at(soo, yoo_np, thr)
    rows.append((C, float(f_in), float(f_oo)))
    print(C, round(f_in,4), round(f_oo,4))

mean_in=np.mean([r[1] for r in rows]); mean_oo=np.mean([r[2] for r in rows])
print("MEAN_IN", round(mean_in,4), "MEAN_OOD", round(mean_oo,4))
import json
print("JSON", json.dumps({"in":mean_in,"ood":mean_oo,"rows":rows}))
