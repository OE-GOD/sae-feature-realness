"""FIRST-PRINCIPLES: is the model's thinking COMPOSITIONAL? Test negation. If 'not good' is built
from 'good' by a reusable operation, then (a) negating flips where a phrase lands on the valence
direction, and (b) the negation SHIFT is consistent across phrases (a single 'not' operation).
If negation is memorized per-phrase, neither holds."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from datasets import load_dataset
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
L=9; HOOK=f"blocks.{L}.hook_resid_post"
def resid(text):
    toks=model.to_tokens(text)
    with torch.no_grad(): _,c=model.run_with_cache(toks,names_filter=[HOOK])
    return c[HOOK][0].float().mean(0)
# valence direction from SST-2
ds=load_dataset("stanfordnlp/sst2",split="train")
pos=[ds[i]["sentence"] for i in range(1500) if ds[i]["label"]==1][:40]; neg=[ds[i]["sentence"] for i in range(1500) if ds[i]["label"]==0][:40]
val=(torch.stack([resid(t) for t in pos]).mean(0)-torch.stack([resid(t) for t in neg]).mean(0)); val=val/val.norm()
POSADJ=["good","great","wonderful","excellent","fantastic","amazing","brilliant","lovely","superb","delightful"]
NEGADJ=["bad","terrible","awful","horrible","dreadful","disappointing","boring","poor","lousy","mediocre"]
def vscore(adj,negate): return float(resid(f"The movie was {'not ' if negate else ''}{adj}.")@val)
ap=[vscore(a,False) for a in POSADJ]; npg=[vscore(a,True) for a in POSADJ]
an=[vscore(a,False) for a in NEGADJ]; nng=[vscore(a,True) for a in NEGADJ]
print(f"valence projection (higher = more positive):")
print(f"  affirmative positive ('good')      : {np.mean(ap):+.2f}")
print(f"  NEGATED  positive   ('not good')   : {np.mean(npg):+.2f}   (compositional => should DROP)")
print(f"  affirmative negative ('bad')       : {np.mean(an):+.2f}")
print(f"  NEGATED  negative   ('not bad')    : {np.mean(nng):+.2f}   (compositional => should RISE)")
flip_pos=np.mean(np.array(npg)<np.array(ap)); flip_neg=np.mean(np.array(nng)>np.array(an))
print(f"\n  'not {{pos}}' lands below '{{pos}}': {flip_pos:.0%} of phrases | 'not {{neg}}' above '{{neg}}': {flip_neg:.0%}")
shift_pos=np.array(npg)-np.array(ap); shift_neg=np.array(nng)-np.array(an)
print(f"  mean negation shift: positives {shift_pos.mean():+.2f}, negatives {shift_neg.mean():+.2f}  (consistent op => similar magnitude, opposite-correcting)")
# is there a SINGLE negation direction? shift vectors in resid space, cosine consistency
sv=[resid(f"The movie was not {a}.")-resid(f"The movie was {a}.") for a in POSADJ+NEGADJ]
sv=torch.stack(sv); svn=sv/sv.norm(dim=1,keepdim=True); cos=(svn@svn.T); offdiag=cos[~torch.eye(len(sv),dtype=bool)]
print(f"  negation-shift vectors avg pairwise cosine = {offdiag.mean():.2f}  (near 1 => one reusable 'not' operation; near 0 => idiosyncratic)")
