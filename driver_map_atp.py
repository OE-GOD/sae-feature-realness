"""FULL driver-map via attribution patching: one backward pass gives a CAUSE-score for ALL 16k SAE
features at once (grad of the sentiment pos-neg logit w.r.t. each feature's activation x activation).
Map read-score (correlation) vs cause-score (AtP). Validate AtP against the steering-confirmed
drivers (8000, 14733, 8836, 7234 from the prototype)."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.float32)  # fp32 for clean grads
L=12; HOOK=f"blocks.{L}.hook_resid_post"; sid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item()
POS,NEG=sid(" positive"),sid(" negative")
FEW=("Review: A wonderful, heartwarming film.\nSentiment: positive\n"
     "Review: A boring, pointless waste of time.\nSentiment: negative\n")
store={}
def splice(resid, hook):
    feat=sae.encode(resid); feat.retain_grad(); store['f']=feat
    recon=sae.decode(feat); return recon+(resid-recon).detach()
def attribution(text):
    toks=model.to_tokens(FEW+f"Review: {text}\nSentiment:")
    model.zero_grad(set_to_none=True)
    logits=model.run_with_hooks(toks,fwd_hooks=[(HOOK,splice)])
    M=logits[0,-1,POS]-logits[0,-1,NEG]; M.backward()
    f=store['f']; return (f.grad[0]*f[0]).sum(0).detach().cpu().numpy()   # [16384] summed over positions

ds=load_dataset("stanfordnlp/sst2",split="train")
test=[ds[i]["sentence"] for i in range(600) if ds[i]["label"]==1][:14]+[ds[i]["sentence"] for i in range(600) if ds[i]["label"]==0][:14]
cause=np.mean([attribution(t) for t in test],axis=0)              # AtP cause-score per feature

d=torch.load("sem_pooling_dataset.pt", weights_only=False)
X=d["train"][0]["max"].float().numpy(); y=d["train"][1].numpy(); yc=y-y.mean(); zc=X-X.mean(0)
read=np.nan_to_num((zc*yc[:,None]).mean(0)/(zc.std(0)*yc.std()+1e-9)); alive=np.where((X>0).mean(0)>0.02)[0]
ac=np.abs(cause); ar=np.abs(read)
print("VALIDATION — do steering-confirmed drivers rank high in AtP |cause|? (rank among alive feats, lower=stronger)")
order=alive[np.argsort(-ac[alive])]; rankof={int(f):i for i,f in enumerate(order)}
for f in [8000,14733,8836,7234,13367,9084,10511]:
    print(f"  feat {f:5d}: |read|={ar[f]:.2f} AtP|cause|={ac[f]:.3f}  cause_rank={rankof.get(f,'?')}/{len(alive)}")
print(f"\nATLAS over {len(alive)} alive features:")
print(f"  corr(|read|,|cause|) = {np.corrcoef(ar[alive],ac[alive])[0,1]:+.3f}")
topread=alive[np.argsort(-ar[alive])[:100]]; topcause=alive[np.argsort(-ac[alive])[:100]]
print(f"  of top-100 READABLE feats, how many are in top-100 by CAUSE (drivers)? {len(set(topread)&set(topcause))}/100")
print(f"  -> the rest are thermometers (readable, not used). overlap low = read can't find the levers.")
