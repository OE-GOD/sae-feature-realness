"""Validated detector for a SAFETY-RELEVANT semantic concept: hate speech.
Sequence-level (pool SAE features over text). SAE-feature probe vs raw baseline."""
import os, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
# safety dataset (fallback chain)
try:
    ds=load_dataset("tweet_eval","hate",split="train"); TXT="text"; LAB="label"; name="tweet_eval/hate"
except Exception:
    ds=load_dataset("stanfordnlp/sst2",split="train"); TXT="sentence"; LAB="label"; name="sst2/sentiment"
print(f"dataset: {name}")

sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)

# balanced sample
pos=[i for i in range(len(ds)) if ds[i][LAB]==1][:250]
neg=[i for i in range(len(ds)) if ds[i][LAB]==0][:250]
idxs=pos+neg
def feat(i):
    ids=model.to_tokens(ds[i][TXT][:300])
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=[hook])
        resid=c[hook][0].float().cpu()
        feats=sae.encode(c[hook][0]).float().cpu()
    if dev=="mps": torch.mps.empty_cache()
    return resid.mean(0), feats.mean(0)        # POOL over sequence (mean)
R=[]; S=[]; y=[]
for n,i in enumerate(idxs):
    r,s=feat(i); R.append(r); S.append(s); y.append(float(ds[i][LAB]))
    if (n+1)%100==0: print(f"  {n+1}/{len(idxs)}")
R=torch.stack(R); S=torch.stack(S); y=torch.tensor(y)
perm=torch.randperm(len(y)); R,S,y=R[perm],S[perm],y[perm]
ntr=int(len(y)*0.7)

def lr_fit(X,y,steps=500,wd=1e-3):
    mu,sd=X.mean(0),X.std(0)+1e-6; Xn=(X-mu)/sd
    w=torch.zeros(X.shape[1],requires_grad=True); b=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([w,b],0.05,weight_decay=wd)
    for _ in range(steps):
        l=torch.nn.functional.binary_cross_entropy_with_logits(Xn@w+b,y); opt.zero_grad(); l.backward(); opt.step()
    return w.detach(),b.detach(),mu,sd
def acc_f1(X,y,w,b,mu,sd):
    p=(((X-mu)/sd)@w+b>0).float(); acc=(p==y).float().mean().item()
    tp=(p*y).sum();fp=(p*(1-y)).sum();fn=((1-p)*y).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return acc,(2*pr*rc/(pr+rc+1e-9)).item()

# SAE-feature probe (sparse top-50)
corr=torch.tensor([((S[:ntr,j]-S[:ntr,j].mean())*(y[:ntr]-y[:ntr].mean())).mean()/(S[:ntr,j].std()+1e-9) for j in range(S.shape[1])])
sel=corr.abs().topk(50).indices
w,b,mu,sd=lr_fit(S[:ntr][:,sel],y[:ntr]); sa,sf=acc_f1(S[ntr:][:,sel],y[ntr:],w,b,mu,sd)
# Raw pooled probe (full 2304)
wr,br,mur,sdr=lr_fit(R[:ntr],y[:ntr],steps=700); ra,rf=acc_f1(R[ntr:],y[ntr:],wr,br,mur,sdr)

print(f"\n=== SAFETY DETECTOR: {name} (held-out, n={len(y)-ntr}) ===")
print(f"{'PROBE':28} {'acc':>6} {'F1':>6}")
print(f"{'SAE-feature (sparse 50)':28} {sa:>6.2f} {sf:>6.2f}")
print(f"{'Raw residual (full 2304)':28} {ra:>6.2f} {rf:>6.2f}")
print(f"\nverdict: {'detector works (F1>0.7)' if sf>0.7 else 'not reliable yet'}; SAE vs raw: {'match' if abs(sf-rf)<0.05 else 'SAE better' if sf>rf else 'raw better'}")
