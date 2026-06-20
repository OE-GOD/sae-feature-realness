import torch
import torch.nn as nn
import numpy as np

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
train_F = d["train_F"].float()
test_F = d["test_F"].float()
ood_F = d["ood_F"].float()
train_L = d["train_L"]; test_L = d["test_L"]; ood_L = d["ood_L"]

N_FEATURES = 30
torch.manual_seed(0); np.random.seed(0)

def f1_score(y, p):
    tp = ((p==1)&(y==1)).sum(); fp=((p==1)&(y==0)).sum(); fn=((p==0)&(y==1)).sum()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

def best_thresh(scores, y):
    order = np.argsort(scores)
    cand = np.unique(scores)
    cand = np.concatenate([[cand[0]-1e-6],(cand[:-1]+cand[1:])/2,[cand[-1]+1e-6]]) if len(cand)>1 else np.array([cand[0]-1e-6,cand[0]+1e-6])
    bestf,bestt=-1,0.0
    for t in cand:
        p=(scores>=t).astype(int)
        f=f1_score(y,p)
        if f>bestf: bestf,bestt=f,t
    return bestt

def meandiff_select(F, y, k):
    Fn = F.numpy(); yn = y.numpy().astype(bool)
    m1 = Fn[yn].mean(0); m0 = Fn[~yn].mean(0)
    diff = np.abs(m1-m0)
    return np.argsort(-diff)[:k]

class MLP(nn.Module):
    def __init__(self,din):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(din,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

results=[]
indist_f1s=[]; ood_f1s=[]

for C in keys:
    ytr = train_L[C].float()
    if ytr.sum() < 5:
        continue
    cols = meandiff_select(train_F, ytr, N_FEATURES)
    Xtr = train_F[:, cols]
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1.0
    Xtr = (Xtr-mu)/sd
    Xte = (test_F[:,cols]-mu)/sd
    Xood = (ood_F[:,cols]-mu)/sd

    model = MLP(N_FEATURES)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for ep in range(300):
        opt.zero_grad()
        out = model(Xtr)
        loss = lossf(out, ytr)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        str_ = torch.sigmoid(model(Xtr)).numpy()
        ste = torch.sigmoid(model(Xte)).numpy()
        sood = torch.sigmoid(model(Xood)).numpy()
    t = best_thresh(str_, ytr.numpy().astype(int))
    yte = test_L[C].numpy().astype(int)
    yood = ood_L[C].numpy().astype(int)
    f_te = f1_score(yte, (ste>=t).astype(int))
    f_ood = f1_score(yood, (sood>=t).astype(int))
    results.append((C, float(f_te), float(f_ood)))
    indist_f1s.append(f_te); ood_f1s.append(f_ood)
    print(f"{C}: indist={f_te:.4f} ood={f_ood:.4f}")

print("MEAN_INDIST", float(np.mean(indist_f1s)))
print("MEAN_OOD", float(np.mean(ood_f1s)))
import json
print("JSON", json.dumps(results))
