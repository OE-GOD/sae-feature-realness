import torch, numpy as np, json
torch.manual_seed(0); np.random.seed(0)

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
trF = d["train_F"].float(); teF = d["test_F"].float(); ooF = d["ood_F"].float()

NF=30

def select_corr(F, y, k):
    yc = y - y.mean()
    Fc = F - F.mean(0)
    num = (Fc * yc[:,None]).sum(0)
    den = torch.sqrt((Fc**2).sum(0) * (yc**2).sum() + 1e-12)
    r = (num/den).abs()
    return torch.argsort(r, descending=True)[:k]

def f1_np(y, p):
    tp=np.sum((p==1)&(y==1)); fp=np.sum((p==1)&(y==0)); fn=np.sum((p==0)&(y==1))
    if tp==0: return 0.0
    pr=tp/(tp+fp); rc=tp/(tp+fn)
    return 2*pr*rc/(pr+rc)

def best_thresh(probs, y):
    ts=np.unique(probs)
    bf,bt=-1,0.5
    for t in ts:
        f=f1_np(y,(probs>=t).astype(int))
        if f>bf: bf,bt=f,t
    return bt

class MLP(torch.nn.Module):
    def __init__(s,din):
        super().__init__()
        s.net=torch.nn.Sequential(torch.nn.Linear(din,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    def forward(s,x): return s.net(x).squeeze(-1)

rows=[]
for C in keys:
    ytr_t = d["train_L"][C].float()
    if ytr_t.sum().item()<5: continue
    ytr=ytr_t.numpy().astype(int)
    yte=d["test_L"][C].numpy().astype(int)
    yoo=d["ood_L"][C].numpy().astype(int)
    cols=select_corr(trF, ytr_t, NF)
    Xtr=trF[:,cols]; mu=Xtr.mean(0); sd=Xtr.std(0)+1e-8
    Xtr=(Xtr-mu)/sd; Xte=(teF[:,cols]-mu)/sd; Xoo=(ooF[:,cols]-mu)/sd

    torch.manual_seed(0)
    model=MLP(NF)
    opt=torch.optim.Adam(model.parameters(),lr=1e-2,weight_decay=1e-4)
    lossf=torch.nn.BCEWithLogitsLoss()
    for ep in range(300):
        opt.zero_grad()
        out=model(Xtr)
        loss=lossf(out,ytr_t)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ptr=torch.sigmoid(model(Xtr)).numpy()
        pte=torch.sigmoid(model(Xte)).numpy()
        poo=torch.sigmoid(model(Xoo)).numpy()
    t=best_thresh(ptr,ytr)
    f_te=f1_np(yte,(pte>=t).astype(int))
    f_oo=f1_np(yoo,(poo>=t).astype(int))
    rows.append({"concept":C,"indist_f1":float(f_te),"ood_f1":float(f_oo)})

mte=float(np.mean([r["indist_f1"] for r in rows]))
moo=float(np.mean([r["ood_f1"] for r in rows]))
out={"recipe":"corr|nf30|mlp","mean_indist_f1":mte,"mean_ood_f1":moo,"per_concept":rows}
print(json.dumps(out,indent=2))
