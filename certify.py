"""certify_feature(): the realness certification battery.
A feature is CERTIFIED only if it passes ALL axes that don't overlap:
  STABILITY (structural) + NECESSITY + SUFFICIENCY (causal) + COHERENCE (meaning).
Thresholds are PROVISIONAL (toy-calibrated). Industrial use needs cross-model
validation + labeled threshold calibration (tasks #72, #73)."""
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

# --- PROVISIONAL thresholds (document every one) ---
TAU_STAB=0.70     # cross-seed cosine
TAU_COH =0.50     # top-3 token concentration
TAU_SUFF=0.50     # steering must raise own-token logit by this
# necessity: must beat the control noise band (computed empirically below)

sae=TinySAE(); sae.load_state_dict(torch.load("/Users/oe/rebuild/sae.pt")); sae.eval()
sae2=TinySAE(); sae2.load_state_dict(torch.load("/Users/oe/rebuild/sae2.pt")); sae2.eval()
model=HookedTransformer.from_pretrained("pythia-160m")
data=load_dataset("NeelNanda/pile-10k",split="train")
N=2048; Wd=sae.W_dec.weight.detach()
with torch.no_grad():
    W2=sae2.W_dec.weight
    STAB=((Wd/Wd.norm(dim=0)).T@(W2/W2.norm(dim=0))).max(1).values

# corpus + per-feature interp/freq/top-token
firetok=[Counter() for _ in range(N)]; fires=torch.zeros(N); docs=[]
for i in range(15):
    ids=model.to_tokens(data[i]["text"][:600])
    with torch.no_grad():
        base,c=model.run_with_cache(ids,return_type="loss")
        _,sp=sae(c["blocks.6.hook_resid_post"][0])
    docs.append((ids,base.item(),sp))
    tid=ids[0].tolist(); r,cc=torch.where(sp>0)
    for rr,k in zip(r.tolist(),cc.tolist()): firetok[k][tid[rr]]+=1
    fires+=(sp>0).float().sum(0)
INTERP=torch.zeros(N); TOPTOK=[None]*N
for f in range(N):
    c=firetok[f]; t=sum(c.values())
    if t>0: INTERP[f]=sum(v for _,v in c.most_common(3))/t; TOPTOK[f]=c.most_common(1)[0][0]

# necessity noise band: ablate 15 RANDOM directions, per-firing effect spread
torch.manual_seed(7)
band=[]
for _ in range(15):
    rd=torch.randn(768); rd/=rd.norm()
    tot=0.0; nf=0
    for ids,base,sp in docs:
        amt=sp[:, _ % N]  # arbitrary firing pattern as placeholder magnitude
        contrib=(amt.unsqueeze(1)*rd.unsqueeze(0))
        def hk(rp,hook,c=contrib): rp[0]=rp[0]-c.to(rp.device); return rp
        with torch.no_grad():
            abl=model.run_with_hooks(ids,return_type="loss",fwd_hooks=[("blocks.6.hook_resid_post",hk)])
        tot+=abs(abl.item()-base); nf+=int((amt>0).sum())
    if nf>0: band.append(tot/nf)
NOISE=sorted(band)[len(band)//2]*2  # 2x median random effect = the bar
print(f"necessity noise band (must exceed): {NOISE:.5f}\n")

def certify(f):
    col=Wd[:,f]; t=TOPTOK[f]
    # necessity
    tot=0.0; nf=0
    for ids,base,sp in docs:
        contrib=(sp[:,f].unsqueeze(1)*col.unsqueeze(0))
        def hk(rp,hook,c=contrib): rp[0]=rp[0]-c.to(rp.device); return rp
        with torch.no_grad():
            abl=model.run_with_hooks(ids,return_type="loss",fwd_hooks=[("blocks.6.hook_resid_post",hk)])
        tot+=abs(abl.item()-base); nf+=int((sp[:,f]>0).sum())
    nec=tot/nf if nf else 0
    # sufficiency
    dl=0.0; nd=0
    for ids,base,sp in docs:
        def hk2(rp,hook,c=col): rp[0,-1]=rp[0,-1]+4.0*c.to(rp.device); return rp
        with torch.no_grad():
            l0=model(ids)[0,-1]; l1=model.run_with_hooks(ids,fwd_hooks=[("blocks.6.hook_resid_post",hk2)])[0,-1]
        dl+=(l1[t]-l0[t]).item(); nd+=1
    suf=dl/nd if nd else 0
    card={"stability":(STAB[f].item(),STAB[f].item()>=TAU_STAB),
          "coherence":(INTERP[f].item(),INTERP[f].item()>=TAU_COH),
          "necessity":(nec,nec>=NOISE),
          "sufficiency":(suf,suf>=TAU_SUFF)}
    certified=all(p for _,p in card.values())
    return card,certified

alive=torch.where(fires>=10)[0]
sample=alive[torch.linspace(0,len(alive)-1,12).long()].tolist()
n_cert=0
for f in sample:
    card,ok=certify(f)
    if ok: n_cert+=1
    tt=repr(model.tokenizer.decode([TOPTOK[f]])) if TOPTOK[f] is not None else "?"
    line=" | ".join(f"{k}:{'PASS' if p else 'fail'}({v:+.2f})" for k,(v,p) in card.items())
    print(f"f{f:<5} {tt:<10} {'*** CERTIFIED ***' if ok else 'rejected':<18} {line}")
print(f"\n{n_cert}/{len(sample)} features CERTIFIED real (provisional thresholds)")
