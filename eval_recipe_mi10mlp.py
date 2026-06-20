import torch, numpy as np, json

d = torch.load("/Users/oe/rebuild/detector_dataset.pt")
keys = d["keys"]
trF = d["train_F"].float().numpy(); teF = d["test_F"].float().numpy(); ooF = d["ood_F"].float().numpy()

NF=10
torch.manual_seed(0)

def f1(y, pred):
    y=y.astype(int); pred=pred.astype(int)
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    if tp==0: return 0.0
    p=tp/(tp+fp); r=tp/(tp+fn)
    return 2*p*r/(p+r)

def select_mi(F, y, k):
    # binarize feature >0, mutual information from 2x2 contingency
    n=len(y); yb=y.astype(bool)
    scores=np.zeros(F.shape[1])
    for j in range(F.shape[1]):
        xb=F[:,j]>0
        mi=0.0
        for xv in (False,True):
            for yv in (False,True):
                joint=((xb==xv)&(yb==yv)).sum()/n
                if joint<=0: continue
                px=(xb==xv).sum()/n; py=(yb==yv).sum()/n
                mi+=joint*np.log(joint/(px*py))
        scores[j]=mi
    return np.argsort(-scores)[:k]

def best_thresh(probs, y):
    ts=np.unique(probs); bestf,bestt=-1,0.5
    for t in ts:
        f=f1(y,(probs>=t).astype(int))
        if f>bestf: bestf,bestt=f,t
    return bestt

rows=[]
for C in keys:
    ytr=d["train_L"][C].numpy().astype(int)
    if ytr.sum()<5: continue
    yte=d["test_L"][C].numpy().astype(int)
    yoo=d["ood_L"][C].numpy().astype(int)
    cols=select_mi(trF, ytr, NF)
    Xtr=trF[:,cols]; mu=Xtr.mean(0); sd=Xtr.std(0)+1e-8
    Xtr=(Xtr-mu)/sd; Xte=(teF[:,cols]-mu)/sd; Xoo=(ooF[:,cols]-mu)/sd

    # 1-hidden-layer MLP (32 ReLU) + output, BCE, light weight decay
    torch.manual_seed(0)
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.float32)
    net=torch.nn.Sequential(torch.nn.Linear(NF,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    opt=torch.optim.Adam(net.parameters(),lr=1e-2,weight_decay=1e-4)
    lossf=torch.nn.BCEWithLogitsLoss()
    for _ in range(500):
        opt.zero_grad()
        out=net(Xt).squeeze(1)
        loss=lossf(out,yt)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        ptr=torch.sigmoid(net(Xt).squeeze(1)).numpy()
        pte=torch.sigmoid(net(torch.tensor(Xte,dtype=torch.float32)).squeeze(1)).numpy()
        poo=torch.sigmoid(net(torch.tensor(Xoo,dtype=torch.float32)).squeeze(1)).numpy()
    t=best_thresh(ptr,ytr)
    f_te=f1(yte,(pte>=t).astype(int))
    f_oo=f1(yoo,(poo>=t).astype(int))
    rows.append((C,float(f_te),float(f_oo)))
    print(C,round(f_te,4),round(f_oo,4))

mte=float(np.mean([r[1] for r in rows])); moo=float(np.mean([r[2] for r in rows]))
print("MEAN_INDIST",round(mte,4)); print("MEAN_OOD",round(moo,4))
print("JSON",json.dumps([{"concept":r[0],"indist_f1":r[1],"ood_f1":r[2]} for r in rows]))
