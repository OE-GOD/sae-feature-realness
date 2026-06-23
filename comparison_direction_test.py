"""E6: is comparison a steerable DIRECTION? Build role-direction w_hat = mean[resid('A better than B')
- resid('B better than A')] at the final token (the 'A-wins minus B-wins' direction), same recipe that
gave a clean steerable 'not' (cos 0.92). Measure reusability (cosine) and whether STEERING it flips the
winner. If it can't flip the winner (only shifts global), comparison is 'compositional but NOT a direction'."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
tok1=lambda w: model.to_tokens(" "+w,prepend_bos=False)[0]
cands=["Toyota","Honda","Coke","Pepsi","Apple","Google","Ford","Tesla","Sony","Intel","Amazon","Netflix","Boeing","Visa","Disney","Shell","Adobe","Oracle","Pfizer","Marvel","Lego","Rolex","Nike","Gucci"]
S=[w for w in cands if tok1(w).shape[0]==1]; pairs=[(S[i],S[i+1]) for i in range(0,len(S)-1,2)]
FEW=("Diamond is better than glass. Which is better? Answer: Diamond\n"
     "Rust is worse than gold. Which is better? Answer: gold\n")
L=15; HOOK=f"blocks.{L}.hook_resid_post"
def run(a,b,comp,steer=None):
    toks=model.to_tokens(FEW+f"{a} is {comp} than {b}. Which is better? Answer:")
    hooks=[(HOOK, lambda r,hook:(r[:, -1:, :].add_(steer) or r) if False else torch.cat([r[:,:-1,:], r[:,-1:,:]+steer],1))] if steer is not None else []
    with torch.no_grad(): lg=model.run_with_hooks(toks,fwd_hooks=hooks)
    return (lg[0,-1,tok1(a)[0]]-lg[0,-1,tok1(b)[0]]).item()
def fin(a,b,comp):
    toks=model.to_tokens(FEW+f"{a} is {comp} than {b}. Which is better? Answer:")
    with torch.no_grad(): _,c=model.run_with_cache(toks,names_filter=[HOOK])
    return c[HOOK][0,-1].float()
tr=pairs[:8]
shifts=[fin(a,b,"better")-fin(b,a,"better") for a,b in tr]   # A-wins minus B-wins (winner is mention-1)
w=torch.stack(shifts).mean(0)
sn=torch.stack([s/s.norm() for s in shifts]); cos=(sn@sn.T); off=cos[~torch.eye(len(sn),dtype=bool)]
print(f"comparison role-direction reusability: avg pairwise cosine = {off.mean():.2f}  (negation was 0.92)")
te=pairs[8:]
print(f"\nSTEER 'worse' prompts (baseline B wins, diff<0) with +k*w_hat (should push A to win if it's a direction):")
print(f"{'k':>4}{'mean diff(A-B)':>16}{'A-wins rate':>13}")
for k in [0,2,4,8]:
    sd=(k*w).to(model.cfg.dtype)
    ds=[run(a,b,"worse",steer=sd if k>0 else None) for a,b in te]
    print(f"{k:>4}{np.mean(ds):>16.2f}{np.mean(np.array(ds)>0):>13.0%}")
print("\nif diff stays <0 / A-wins rate ~0 even at high k => steering a direction CANNOT flip the winner")
print("=> comparison is COMPOSITIONAL (E1) but NOT a steerable direction. (low cosine reinforces.)")
