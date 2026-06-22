"""The decisive contrast: is 'truth' a clean linear knob like sentiment, or not? Same protocol:
truth read-direction = difference-of-means of residual between TRUE and FALSE claims; use it as a
write-direction; measure the model's True/False judgment dose-response. If facts lack a clean
direction, the knob is weak/non-monotone vs sentiment's clean one -> our concept/fact split IS the
clean-direction split (first-principles unification)."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from datasets import load_dataset
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
sid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item()
TRUE,FALSE=sid(" True"),sid(" False")
L=9; HOOK=f"blocks.{L}.hook_resid_post"
FEW=("Statement: The capital of Italy is Rome.\nAnswer: True\n"
     "Statement: The sun rises in the west.\nAnswer: False\n")
def judge_logitdiff(claim, steer=None):
    toks=model.to_tokens(FEW+f"Statement: {claim}\nAnswer:")
    hooks=[(HOOK, lambda r,hook: r+steer)] if steer is not None else []
    with torch.no_grad(): logits=model.run_with_hooks(toks,fwd_hooks=hooks)
    return (logits[0,-1,TRUE]-logits[0,-1,FALSE]).item()
def resid_mean(claim):
    toks=model.to_tokens(f"Statement: {claim}")
    with torch.no_grad(): _,c=model.run_with_cache(toks,names_filter=[HOOK])
    return c[HOOK][0].float().mean(0)

ds=load_dataset("notrichardren/azaria-mitchell",split="train")
idx=[i for i in range(len(ds)) if ds[i]["dataset"] in ("cities","companies","inventions")]
T=[ds[i]["claim"] for i in idx if ds[i]["label"]==1][:60]
Fa=[ds[i]["claim"] for i in idx if ds[i]["label"]==0][:60]
mt=torch.stack([resid_mean(t) for t in T[:40]]).mean(0); mf=torch.stack([resid_mean(t) for t in Fa[:40]]).mean(0)
truth_dir=(mt-mf)
rng=torch.Generator(device=dev).manual_seed(0)
rand=torch.randn(truth_dir.shape,generator=rng,device=dev).to(truth_dir.dtype); rand=rand/rand.norm()*truth_dir.norm()
test=T[40:55]+Fa[40:55]
print(f"truth dir ||mt-mf||={truth_dir.norm():.2f}  (alpha=1 adds one full true-vs-false difference)")
print(f"\n{'alpha':>7}{'truth_dir (True-False logit)':>30}{'random_dir (control)':>24}")
for a in [-3,-2,-1,0,1,2,3]:
    sd=(a*truth_dir).to(model.cfg.dtype); rd=(a*rand).to(model.cfg.dtype)
    ms=np.mean([judge_logitdiff(t,steer=sd) for t in test]); mr=np.mean([judge_logitdiff(t,steer=rd) for t in test])
    print(f"{a:>7}{ms:>30.2f}{mr:>24.2f}")
print("\nCompare the SWING to sentiment's. Clean monotone knob => truth is also a clean direction;")
print("weak/flat/non-monotone => facts lack a clean steerable direction (the concept/fact split IS the linearity split).")
