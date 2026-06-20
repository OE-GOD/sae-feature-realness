"""Diagnose WHY hate detection caps at 0.78. Read the failures: are they
ambiguous/mislabeled (label noise) or clear misses (representation failure)?"""
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

pos=[i for i in range(len(ds)) if ds[i]["label"]==1][:300]
neg=[i for i in range(len(ds)) if ds[i]["label"]==0][:300]
idxs=pos+neg
S=[]; y=[]; txt=[]
for n,i in enumerate(idxs):
    ids=model.to_tokens(ds[i]["text"][:300])
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=[hook])
        f=sae.encode(c[hook][0]).float().cpu().mean(0)
    S.append(f); y.append(float(ds[i]["label"])); txt.append(ds[i]["text"])
    if dev=="mps": torch.mps.empty_cache()
S=torch.stack(S); y=torch.tensor(y); txt=list(txt)
g=torch.Generator().manual_seed(0); perm=torch.randperm(len(y),generator=g)
S,y=S[perm],y[perm]; txt=[txt[i] for i in perm.tolist()]; ntr=int(len(y)*0.7)

corr=torch.tensor([((S[:ntr,j]-S[:ntr,j].mean())*(y[:ntr]-y[:ntr].mean())).mean()/(S[:ntr,j].std()+1e-9) for j in range(S.shape[1])])
sel=corr.abs().topk(100).indices
X=S[:,sel]; mu,sd=X[:ntr].mean(0),X[:ntr].std(0)+1e-6; X=(X-mu)/sd
w=torch.zeros(100,requires_grad=True); b=torch.zeros(1,requires_grad=True)
opt=torch.optim.Adam([w,b],0.03,weight_decay=2e-3)
for _ in range(800):
    l=torch.nn.functional.binary_cross_entropy_with_logits(X[:ntr]@w+b,y[:ntr]);opt.zero_grad();l.backward();opt.step()
with torch.no_grad():
    logits=X[ntr:]@w+b; conf=torch.sigmoid(logits); pred=(logits>0).float()
yte=y[ntr:]; tte=txt[ntr:]
errs=[(abs(conf[i].item()-0.5), conf[i].item(), int(yte[i]), int(pred[i]), tte[i]) for i in range(len(yte)) if pred[i]!=yte[i]]
n_err=len(errs); n_total=len(yte)
# how many errors are near the boundary (|conf-0.5|<0.15) vs confident-wrong (>0.35)
near=sum(1 for e in errs if e[0]<0.15); confident=sum(1 for e in errs if e[0]>0.35)
print(f"=== ERROR ANALYSIS: hate detector ({n_err}/{n_total} errors) ===")
print(f"near decision boundary (ambiguous): {near}/{n_err}")
print(f"confidently wrong (systematic):     {confident}/{n_err}")
print("\n--- sample misclassifications (true -> pred, conf) ---")
errs.sort()  # ambiguous first
for _,cf,tr,pr,t in errs[:12]:
    tag="MISS hate" if tr==1 else "false alarm"
    print(f"[{tag}] true={tr} pred={pr} conf={cf:.2f} | {t[:110]}")
