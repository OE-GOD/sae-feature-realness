"""Research: what makes a concept trustworthily buildable? Test if LAYER + POOLING
move hate-speech detection from 0.68 toward certifiable. 2x2: layer{12,22} x pool{mean,last}."""
import os, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae12=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
sae22=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_22/width_16k/average_l0_72",device=dev)
h12="blocks.12.hook_resid_post"; h22="blocks.22.hook_resid_post"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
ds=load_dataset("tweet_eval","hate",split="train")

pos=[i for i in range(len(ds)) if ds[i]["label"]==1][:250]
neg=[i for i in range(len(ds)) if ds[i]["label"]==0][:250]
idxs=pos+neg
# collect 4 representations per example
feats={"L12_mean":[],"L12_last":[],"L22_mean":[],"L22_last":[]}; y=[]
for n,i in enumerate(idxs):
    ids=model.to_tokens(ds[i]["text"][:300])
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=[h12,h22])
        f12=sae12.encode(c[h12][0]).float().cpu()
        f22=sae22.encode(c[h22][0]).float().cpu()
    feats["L12_mean"].append(f12.mean(0)); feats["L12_last"].append(f12[-1])
    feats["L22_mean"].append(f22.mean(0)); feats["L22_last"].append(f22[-1])
    y.append(float(ds[i]["label"]))
    if (n+1)%100==0: print(f"  {n+1}/{len(idxs)}")
    if dev=="mps": torch.mps.empty_cache()
y=torch.tensor(y); perm=torch.randperm(len(y)); ntr=int(len(y)*0.7)

def evaluate(X):
    X=X[perm]
    Xtr,Xte=X[:ntr],X[ntr:]; ytr,yte=y[perm][:ntr],y[perm][ntr:]
    corr=torch.tensor([((Xtr[:,j]-Xtr[:,j].mean())*(ytr-ytr.mean())).mean()/(Xtr[:,j].std()+1e-9) for j in range(X.shape[1])])
    sel=corr.abs().topk(50).indices
    mu,sd=Xtr[:,sel].mean(0),Xtr[:,sel].std(0)+1e-6
    Ztr=(Xtr[:,sel]-mu)/sd
    w=torch.zeros(50,requires_grad=True); b=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([w,b],0.05,weight_decay=1e-3)
    for _ in range(500):
        l=torch.nn.functional.binary_cross_entropy_with_logits(Ztr@w+b,ytr); opt.zero_grad(); l.backward(); opt.step()
    Zte=(Xte[:,sel]-mu)/sd
    with torch.no_grad():
        p=((Zte@w+b)>0).float()
    acc=(p==yte).float().mean().item()
    tp=(p*yte).sum();fp=(p*(1-yte)).sum();fn=((1-p)*yte).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9); return acc,(2*pr*rc/(pr+rc+1e-9)).item()

print(f"\n=== hate-speech detector: layer x pooling (held-out n={len(y)-ntr}) ===")
print(f"{'config':12} {'acc':>6} {'F1':>6}")
best=None
for k in ["L12_mean","L12_last","L22_mean","L22_last"]:
    a,f=evaluate(torch.stack(feats[k]))
    print(f"{k:12} {a:>6.2f} {f:>6.2f}")
    if best is None or f>best[1]: best=(k,f)
print(f"\nbest: {best[0]} F1 {best[1]:.2f}  ({'CERTIFIED' if best[1]>0.8 else 'closer but not certified' if best[1]>0.72 else 'still not reliable'})")
print("(layer-12 mean was 0.68 baseline)")
