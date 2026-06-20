import os, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print("loading 16k SAE (cached)...")
sae16 = SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device="cpu")
print("loading 65k SAE (downloads ~1.2GB once)...")
sae65 = SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_65k/average_l0_72",device="cpu")
hook = sae16.cfg.metadata["hook_name"]

def decmat(sae):
    W = sae.W_dec.detach().float()           # want [n_features, d_in]
    if W.shape[0] != sae.cfg.d_sae: W = W.T
    return W
W16 = decmat(sae16); W65 = decmat(sae65)
print("W16", tuple(W16.shape), "W65", tuple(W65.shape))

# CHEAP signal: firing frequency of 16k features (run model on text)
print("loading gemma-2-2b...")
model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)
data = load_dataset("NeelNanda/pile-10k", split="train")
sae16_dev = sae16.to(dev)
D=sae16.cfg.d_sae; freq=torch.zeros(D); ntok=0
for i in range(40):
    ids = model.to_tokens(data[i]["text"][:500])
    with torch.no_grad():
        _,c = model.run_with_cache(ids, names_filter=[hook])
        f = sae16_dev.encode(c[hook][0]).float().cpu()
    freq += (f>0).float().sum(0); ntok += f.shape[0]
    if dev=="mps": torch.mps.empty_cache()
freq/=ntok
print("tokens:", ntok)

# EXPENSIVE ground truth: cross-width stability (best cosine 16k feature -> any 65k feature)
W16n = W16/ (W16.norm(dim=1,keepdim=True)+1e-9)
W65n = W65/ (W65.norm(dim=1,keepdim=True)+1e-9)
stab=torch.zeros(D)
for i in range(0,D,1024):
    chunk = W16n[i:i+1024] @ W65n.T          # [chunk, 65536]
    stab[i:i+1024] = chunk.max(dim=1).values
print("stability computed")

alive=freq>0
f=freq[alive]; s=stab[alive]; lf=torch.log10(f+1e-9)
def pear(a,b):
    a=a-a.mean(); b=b-b.mean(); return (a@b/(a.norm()*b.norm())).item()
print(f"\n=== REAL Gemma Scope, cross-width 16k vs 65k ===")
print(f"alive features: {int(alive.sum())}/{D}")
print(f"Pearson(log-freq, stability) = {pear(lf,s):.3f}   (toy 0.52, scaled 0.54)")
import torch as T
qs=T.quantile(lf, T.linspace(0,1,5))
print("quartile -> avg stability:")
for i,lab in enumerate(["Q1 rarest","Q2","Q3","Q4 commonest"]):
    m=(lf>=qs[i])&(lf<=qs[i+1]); print(f"  {lab:14s}: {s[m].mean():.3f}")
