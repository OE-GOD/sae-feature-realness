import os, torch, torch.nn as nn
from collections import Counter
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
pile=torch.load("/Users/oe/rebuild/gemma_acts.pt"); DM=pile.shape[1]; N=2048
print("acts", tuple(pile.shape))

class SAE(nn.Module):
    def __init__(s,d,n=N,k=32):
        super().__init__(); s.k=k; s.W_enc=nn.Linear(d,n); s.W_dec=nn.Linear(n,d)
    def forward(s,x):
        sc=s.W_enc(x); tk=torch.topk(sc,s.k,-1)
        sp=torch.zeros_like(sc); sp.scatter_(-1,tk.indices,tk.values)
        return s.W_dec(sp),sp
def train(seed):
    torch.manual_seed(seed); m=SAE(DM); opt=torch.optim.Adam(m.parameters(),1e-3)
    for ep in range(5):
        perm=torch.randperm(len(pile))
        for i in range(0,len(pile),1024):
            b=pile[perm[i:i+1024]]; r,_=m(b); l=((r-b)**2).mean()
            opt.zero_grad(); l.backward(); opt.step()
    return m
saes=[train(s) for s in range(3)]; print("3 SAEs trained")
sae=saes[0]; Wd=sae.W_dec.weight.detach()

cols=[(m.W_dec.weight/m.W_dec.weight.norm(dim=0)).detach() for m in saes]
with torch.no_grad():
    b=[(cols[0].T@cols[j]).max(1).values for j in (1,2)]
    stab=torch.stack(b).median(0).values

print("loading gemma..."); model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
data=load_dataset("NeelNanda/pile-10k",split="train"); HOOK="blocks.12.hook_resid_post"
docs=[]; firetok=[Counter() for _ in range(N)]
for i in range(8):
    ids=model.to_tokens(data[i]["text"][:500])
    with torch.no_grad():
        base,c=model.run_with_cache(ids,return_type="loss",names_filter=[HOOK])
        _,sp=sae(c[HOOK][0].float().cpu())
    docs.append((ids,base.item(),sp))
    tid=ids[0].tolist(); rows,cc=torch.where(sp>0)
    for r,k in zip(rows.tolist(),cc.tolist()): firetok[k][tid[r]]+=1
    if dev=="mps": torch.mps.empty_cache()

interp=torch.zeros(N); fires=torch.zeros(N)
for f in range(N):
    cnt=firetok[f]; tot=sum(cnt.values()); fires[f]=tot
    if tot>0: interp[f]=sum(v for _,v in cnt.most_common(3))/tot

import statistics
def pear(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b)); sa=sum((x-ma)**2 for x in a)**.5; sb=sum((y-mb)**2 for y in b)**.5
    return cov/(sa*sb) if sa*sb else 0

alive=torch.where(fires>=8)[0]
# stability vs interp (cheap, all alive)
si_s=[stab[f].item() for f in alive]; si_i=[interp[f].item() for f in alive]

# sample for causal
sample=alive[interp[alive].argsort()][torch.linspace(0,len(alive)-1,25).long()].tolist()
ci=[]; cc_=[]; cs=[]
for f in sample:
    col=Wd[:,f]; tot=0.0; nf=0
    for ids,base,sp in docs:
        contrib=(sp[:,f].unsqueeze(1)*col.unsqueeze(0))
        def hk(rp,hook,c=contrib): rp[0]=rp[0]-c.to(rp.device); return rp
        with torch.no_grad():
            abl=model.run_with_hooks(ids,return_type="loss",fwd_hooks=[(HOOK,hk)])
        tot+=(abl.item()-base); nf+=int((sp[:,f]>0).sum())
    if nf>0:
        cv=tot/nf; ci.append(interp[f].item()); cc_.append(cv); cs.append(stab[f].item())
    if dev=="mps": torch.mps.empty_cache()

print("\n=== GEMMA-2-2B triangle of 'real' ===")
print(f"stability vs interpretability = {pear(si_s,si_i):+.3f}   (Pythia +0.00)")
print(f"stability vs causation        = {pear(cs,cc_):+.3f}   (Pythia -0.13)")
print(f"interpretability vs causation = {pear(ci,cc_):+.3f}   (Pythia +0.37)")
