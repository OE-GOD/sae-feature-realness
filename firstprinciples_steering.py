"""FIRST-PRINCIPLES test of the Linear Representation Hypothesis on the thing we call a 'clean
direction' (sentiment). Read-direction = difference-of-means of the residual stream between pos/neg.
Then USE it as a write-direction: add alpha*dir to the residual and measure the model's sentiment
readout. If sentiment is a clean linear feature, we get a smooth MONOTONE dose-response (a volume
knob). Random-direction = control. This tests the foundation under all our probing work."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from datasets import load_dataset
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
sid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item()
POS,NEG=sid(" positive"),sid(" negative")
L=9; HOOK=f"blocks.{L}.hook_resid_post"
FEW=("Review: A wonderful, heartwarming film.\nSentiment: positive\n"
     "Review: A boring, pointless waste of time.\nSentiment: negative\n")
def readout_logitdiff(text, steer=None):
    toks=model.to_tokens(FEW+f"Review: {text}\nSentiment:")
    hooks=[(HOOK, lambda r,hook: r+steer)] if steer is not None else []
    with torch.no_grad(): logits=model.run_with_hooks(toks,fwd_hooks=hooks)
    return (logits[0,-1,POS]-logits[0,-1,NEG]).item()

ds=load_dataset("stanfordnlp/sst2",split="train")
pos=[ds[i]["sentence"] for i in range(2000) if ds[i]["label"]==1][:60]
neg=[ds[i]["sentence"] for i in range(2000) if ds[i]["label"]==0][:60]
def resid_mean(text):
    toks=model.to_tokens(text[:200])
    with torch.no_grad(): _,c=model.run_with_cache(toks,names_filter=[HOOK])
    return c[HOOK][0].float().mean(0)
mp=torch.stack([resid_mean(t) for t in pos[:40]]).mean(0)
mn=torch.stack([resid_mean(t) for t in neg[:40]]).mean(0)
read_dir=(mp-mn)                                   # difference-of-means sentiment direction
rng=torch.Generator(device=dev).manual_seed(0)
rand_dir=torch.randn(read_dir.shape,generator=rng,device=dev).to(read_dir.dtype); rand_dir=rand_dir/rand_dir.norm()*read_dir.norm()
test=pos[40:55]+neg[40:55]                         # held-out reviews
print(f"sentiment dir ||mp-mn||={read_dir.norm():.2f}  (alpha=1 adds one full class-difference)")
print(f"\n{'alpha':>7}{'sentiment_dir (pos-neg logit)':>32}{'random_dir (control)':>24}")
for a in [-3,-2,-1,0,1,2,3]:
    sd=(a*read_dir).to(model.cfg.dtype); rd=(a*rand_dir).to(model.cfg.dtype)
    ms=np.mean([readout_logitdiff(t,steer=sd) for t in test])
    mr=np.mean([readout_logitdiff(t,steer=rd) for t in test])
    print(f"{a:>7}{ms:>32.2f}{mr:>24.2f}")
print("\nMonotone, smooth, proportional in the sentiment column = clean linear 'volume knob'.")
print("Random column flat = the effect is specific to the direction, not generic perturbation.")
