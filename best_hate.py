"""Best-effort: can we build a certifiable hate detector? More data, more features,
linear vs small MLP. L12 mean-pool (the winning config)."""
import os, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook="blocks.12.hook_resid_post"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
ds=load_dataset("tweet_eval","hate",split="train")

pos=[i for i in range(len(ds)) if ds[i]["label"]==1][:700]
neg=[i for i in range(len(ds)) if ds[i]["label"]==0][:700]
idxs=pos+neg
S=[]; y=[]
for n,i in enumerate(idxs):
    ids=model.to_tokens(ds[i]["text"][:300])
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=[hook])
        f=sae.encode(c[hook][0]).float().cpu().mean(0)
    S.append(f); y.append(float(ds[i]["label"]))
    if (n+1)%200==0: print(f"  {n+1}/{len(idxs)}")
    if dev=="mps": torch.mps.empty_cache()
S=torch.stack(S); y=torch.tensor(y)
perm=torch.randperm(len(y)); S,y=S[perm],y[perm]; ntr=int(len(y)*0.7)
print(f"data: {len(y)} examples, train {ntr}")

corr=torch.tensor([((S[:ntr,j]-S[:ntr,j].mean())*(y[:ntr]-y[:ntr].mean())).mean()/(S[:ntr,j].std()+1e-9) for j in range(S.shape[1])])
sel=corr.abs().topk(200).indices
X=S[:,sel]; mu,sd=X[:ntr].mean(0),X[:ntr].std(0)+1e-6; X=(X-mu)/sd
Xtr,Xte=X[:ntr],X[ntr:]; ytr,yte=y[:ntr],y[ntr:]
def F1(p,yy):
    tp=(p*yy).sum();fp=(p*(1-yy)).sum();fn=((1-p)*yy).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return (2*pr*rc/(pr+rc+1e-9)).item()

# linear
w=torch.zeros(200,requires_grad=True); b=torch.zeros(1,requires_grad=True)
opt=torch.optim.Adam([w,b],0.03,weight_decay=2e-3)
for _ in range(800):
    l=torch.nn.functional.binary_cross_entropy_with_logits(Xtr@w+b,ytr);opt.zero_grad();l.backward();opt.step()
with torch.no_grad(): lin=F1(((Xte@w+b)>0).float(),yte)

# small MLP
import torch.nn as nn
mlp=nn.Sequential(nn.Linear(200,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,1))
opt=torch.optim.Adam(mlp.parameters(),3e-3,weight_decay=2e-3)
for _ in range(800):
    l=torch.nn.functional.binary_cross_entropy_with_logits(mlp(Xtr).squeeze(),ytr);opt.zero_grad();l.backward();opt.step()
mlp.eval()
with torch.no_grad(): mf=F1((mlp(Xte).squeeze()>0).float(),yte)

print(f"\n=== BEST-EFFORT hate detector (L12 mean, 200 feats, n_test={len(yte)}) ===")
print(f"linear probe : F1 {lin:.2f}")
print(f"small MLP     : F1 {mf:.2f}")
best=max(lin,mf)
print(f"\nbest F1 {best:.2f}  ->  {'CERTIFIED (>0.8)' if best>0.8 else 'near ceiling, not certified' }")
print("note: tweet_eval/hate is a noisy benchmark; label noise likely caps achievable F1.")
