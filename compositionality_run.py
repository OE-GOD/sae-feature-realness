"""Run the two decisive compositionality probes. (1) NEGATION DRIVER: steer with the 'not' direction
-- does it flip the model's sentiment JUDGMENT, and asymmetrically (positives flip, negatives barely)?
(2) INTENSIFICATION read: does 'very' amplify BOTH poles (very good more positive, very bad more
negative) or only positives? If only positives -> the signed-axis asymmetry is OPERATION-GENERAL."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from datasets import load_dataset
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
L=9; HOOK=f"blocks.{L}.hook_resid_post"; sid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item()
POS,NEG=sid(" positive"),sid(" negative")
FEW=("Review: A wonderful, heartwarming film.\nSentiment: positive\nReview: A boring waste of time.\nSentiment: negative\n")
def resid(text):
    with torch.no_grad(): _,c=model.run_with_cache(model.to_tokens(text),names_filter=[HOOK])
    return c[HOOK][0].float().mean(0)
def judge(text, steer=None):
    hooks=[(HOOK, lambda r,hook: r+steer)] if steer is not None else []
    with torch.no_grad(): lg=model.run_with_hooks(model.to_tokens(FEW+f"Review: {text}\nSentiment:"),fwd_hooks=hooks)
    return (lg[0,-1,POS]-lg[0,-1,NEG]).item()
ds=load_dataset("stanfordnlp/sst2",split="train")
sp=[ds[i]["sentence"] for i in range(1500) if ds[i]["label"]==1][:40]; sn=[ds[i]["sentence"] for i in range(1500) if ds[i]["label"]==0][:40]
val=(torch.stack([resid(t) for t in sp]).mean(0)-torch.stack([resid(t) for t in sn]).mean(0)); val=val/val.norm()
PTR=["good","great","wonderful","excellent","fantastic"]; PTE=["brilliant","lovely","superb","delightful","marvelous"]; NEGA=["bad","terrible","awful","horrible","dreadful"]
n_hat=torch.stack([resid(f"The movie was not {a}.")-resid(f"The movie was {a}.") for a in PTR]).mean(0)
rng=torch.Generator(device=dev).manual_seed(0); rand=torch.randn(n_hat.shape,generator=rng,device=dev).to(n_hat.dtype); rand=rand/rand.norm()*n_hat.norm()
print(f"cos(n_hat, valence) = {float((n_hat/n_hat.norm())@val):+.2f}  (low => 'not' is its own op, not just -valence)")
print("\n(1) NEGATION DRIVER — steer 'The movie was {adj}.' (held-out), sentiment logit (pos-neg):")
print(f"{'k':>3}{'POS adj':>10}{'NEG adj':>10}{'rand POS':>10}")
for k in [0,1,2,3]:
    sd=(k*n_hat).to(model.cfg.dtype); rd=(k*rand).to(model.cfg.dtype)
    mp=np.mean([judge(f"The movie was {a}.",steer=sd) for a in PTE]); mn=np.mean([judge(f"The movie was {a}.",steer=sd) for a in NEGA]); mr=np.mean([judge(f"The movie was {a}.",steer=rd) for a in PTE])
    print(f"{k:>3}{mp:>10.2f}{mn:>10.2f}{mr:>10.2f}")
print("  driver+asymmetric => POS column drops/flips, NEG barely moves, rand flat")
print("\n(2) INTENSIFICATION read — valence projection, does 'very' amplify BOTH poles?")
print(f"{'adj':>10}{'bare':>8}{'very':>8}{'extremely':>11}{'effect':>9}")
for a in PTE+NEGA:
    b=float(resid(f'The movie was {a}.')@val); v=float(resid(f'The movie was very {a}.')@val); e=float(resid(f'The movie was extremely {a}.')@val)
    print(f"{a:>10}{b:>8.2f}{v:>8.2f}{e:>11.2f}{v-b:>+9.2f}")
print("  true gain => positives go UP (+), negatives go DOWN (-). If negatives DON'T go down => asymmetry is OPERATION-GENERAL.")
