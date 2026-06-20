import torch, torch.nn as nn
from collections import Counter
from transformer_lens import HookedTransformer
from datasets import load_dataset

class TinySAE(nn.Module):
    def __init__(s,d=768,n=2048,k=32):
        super().__init__(); s.k=k; s.W_enc=nn.Linear(d,n); s.W_dec=nn.Linear(n,d)
    def forward(s,x):
        sc=s.W_enc(x); tk=torch.topk(sc,s.k,-1)
        sp=torch.zeros_like(sc); sp.scatter_(-1,tk.indices,tk.values)
        return s.W_dec(sp),sp

sae1=TinySAE(); sae1.load_state_dict(torch.load("/Users/oe/rebuild/sae.pt")); sae1.eval()
sae2=TinySAE(); sae2.load_state_dict(torch.load("/Users/oe/rebuild/sae2.pt")); sae2.eval()
model=HookedTransformer.from_pretrained("pythia-160m")
data=load_dataset("NeelNanda/pile-10k",split="train")

# STABILITY (have it)
with torch.no_grad():
    W1=sae1.W_dec.weight; W2=sae2.W_dec.weight
    stab=((W1/W1.norm(dim=0)).T@(W2/W2.norm(dim=0))).max(1).values

# collect (token_id, firing) over ~100 docs
N=2048
firetokens=[Counter() for _ in range(N)]
freq=torch.zeros(N); ntok=0
for i in range(100):
    ids=model.to_tokens(data[i]["text"][:1200])
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=["blocks.6.hook_resid_post"])
        _,sp=sae1(c["blocks.6.hook_resid_post"][0])
    tid=ids[0].tolist()
    nz=(sp>0)
    freq+=nz.float().sum(0); ntok+=sp.shape[0]
    rows,cols=torch.where(nz)
    for r,cc in zip(rows.tolist(),cols.tolist()):
        firetokens[cc][tid[r]]+=1
freq/=ntok

# INTERPRETABILITY = share of firings on top-3 tokens
interp=torch.zeros(N); total_fires=torch.zeros(N)
for f in range(N):
    c=firetokens[f]; tot=sum(c.values()); total_fires[f]=tot
    if tot>0:
        top3=sum(v for _,v in c.most_common(3))
        interp[f]=top3/tot

alive=total_fires>=10
I=interp[alive]; S=stab[alive]; F=torch.log10(freq[alive]+1e-9)
def pear(a,b): a=a-a.mean(); b=b-b.mean(); return (a@b/(a.norm()*b.norm())).item()
print(f"alive features (>=10 fires): {int(alive.sum())}")
print(f"Pearson(interpretability, STABILITY) = {pear(I,S):.3f}   <- your prediction: ~0 (independent)")
print(f"Pearson(interpretability, log-freq)  = {pear(I,F):.3f}   (bonus)")
print(f"\nmean interpretability of STABLE features (cos>0.7): {I[S>0.7].mean():.3f}")
print(f"mean interpretability of UNSTABLE features (cos<0.4): {I[S<0.4].mean():.3f}")

# show a couple examples
def toks(f,n=4):
    return [repr(model.tokenizer.decode([t])) for t,_ in firetokens[f].most_common(n)]
order=torch.where(alive)[0]
hi=order[interp[order].argmax()].item(); lo=order[interp[order].argmin()].item()
print(f"\nMOST interpretable f{hi}: interp {interp[hi]:.2f}, stab {stab[hi]:.2f}, tokens {toks(hi)}")
print(f"LEAST interpretable f{lo}: interp {interp[lo]:.2f}, stab {stab[lo]:.2f}, tokens {toks(lo)}")
