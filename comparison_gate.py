"""E1 behavioral gate: does COMPARISON bind roles, or is it a position heuristic? Same entities/
positions, flip only the comparator. binding => 'better' picks A, 'worse' picks B (winner flips).
recency/first-mention => no flip."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
tok1=lambda w: model.to_tokens(" "+w,prepend_bos=False)[0]
cands=["Toyota","Honda","Coke","Pepsi","Apple","Google","Ford","Tesla","Sony","Intel","Amazon","Netflix",
       "Boeing","Visa","Disney","Shell","Adobe","Oracle","Pfizer","Marvel","Lego","Rolex","Nike","Gucci"]
single=[w for w in cands if tok1(w).shape[0]==1]
pairs=[(single[i],single[i+1]) for i in range(0,len(single)-1,2)]
FEW=("Diamond is better than glass. Which is better? Answer: Diamond\n"
     "Rust is worse than gold. Which is better? Answer: gold\n")
def diff(a,b,comp):  # logit(a)-logit(b) at the answer position
    toks=model.to_tokens(FEW+f"{a} is {comp} than {b}. Which is better? Answer:")
    with torch.no_grad(): lg=model(toks)
    return (lg[0,-1,tok1(a)[0]]-lg[0,-1,tok1(b)[0]]).item()
print(f"{len(pairs)} single-token pairs. logit(A)-logit(B) at answer:")
print(f"{'A':>9}{'B':>9}{'better':>9}{'worse':>9}{'flip(b-w)':>11}")
bet=[];wor=[]
for a,b in pairs:
    db=diff(a,b,"better"); dw=diff(a,b,"worse"); bet.append(db); wor.append(dw)
    print(f"{a:>9}{b:>9}{db:>9.2f}{dw:>9.2f}{db-dw:>11.2f}")
bet=np.array(bet); wor=np.array(wor)
print(f"\nmean: better {bet.mean():+.2f} (binding=> >0, A wins), worse {wor.mean():+.2f} (binding=> <0, B wins)")
print(f"FLIP (better-worse) = {np.mean(bet-wor):+.2f}; sign-flip rate = {np.mean((bet>0)&(wor<0)):.0%}")
print("strong positive flip + high flip-rate => BINDING (role tracked, positions frozen). ~0 => position heuristic.")
