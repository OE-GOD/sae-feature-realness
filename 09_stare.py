import torch, torch.nn as nn
from transformer_lens import HookedTransformer
from datasets import load_dataset

class TinySAE(nn.Module):
    def __init__(self, d_model=768, n_features=2048, k=32):
        super().__init__(); self.k=k
        self.W_enc=nn.Linear(d_model,n_features); self.W_dec=nn.Linear(n_features,d_model)
    def forward(self,x):
        s=self.W_enc(x); tk=torch.topk(s,self.k,dim=-1)
        sp=torch.zeros_like(s); sp.scatter_(-1,tk.indices,tk.values)
        return self.W_dec(sp), sp

sae1=TinySAE(); sae1.load_state_dict(torch.load("/Users/oe/rebuild/sae.pt")); sae1.eval()
sae2=TinySAE(); sae2.load_state_dict(torch.load("/Users/oe/rebuild/sae2.pt")); sae2.eval()
model=HookedTransformer.from_pretrained("pythia-160m")
data=load_dataset("NeelNanda/pile-10k", split="train")
pile=torch.load("/Users/oe/rebuild/thoughts.pt")

with torch.no_grad():
    freq=torch.zeros(2048)
    for i in range(0,len(pile),4096):
        _,sp=sae1(pile[i:i+4096]); freq+=(sp>0).float().sum(0)
    freq/=len(pile)
    W1=sae1.W_dec.weight; W2=sae2.W_dec.weight
    stab=((W1/W1.norm(dim=0)).T @ (W2/W2.norm(dim=0))).max(dim=1).values

# collect contexts from 50 docs
S=[]; CTX=[]
for i in range(50):
    toks=model.to_str_tokens(data[i]["text"][:1500])
    ids=model.to_tokens(data[i]["text"][:1500])
    with torch.no_grad():
        _,c=model.run_with_cache(ids, return_type="loss")
        _,sp=sae1(c["blocks.6.hook_resid_post"][0])
    S.append(sp)
    for p,t in enumerate(toks):
        CTX.append(("".join(toks[max(0,p-4):p])+"«"+t+"»"+"".join(toks[p+1:p+3])).replace("\n","/"))
S=torch.cat(S)
seen=(S>0).float().sum(0)   # fires in THIS corpus

def show(label, fidx):
    print(f"\n=== {label}: feature {fidx}  (freq {freq[fidx]:.4f}, stability {stab[fidx]:.2f}) ===")
    vals,idx=S[:,fidx].topk(8)
    for v,j in zip(vals,idx):
        if v.item()==0: break
        print(f"   {v.item():6.2f}  {CTX[j][:80]}")

cand=torch.where(seen>=5)[0]
# commonest (Q4): highest frequency
common=cand[freq[cand].argmax()].item()
# sweet spot (Q3): highest stability among mid-high freq
mid=cand[(freq[cand]>freq.median())]
sweet=mid[stab[mid].argmax()].item()
# rarest (Q1): lowest frequency that still fires >=5 times
rare=cand[freq[cand].argmin()].item()

show("COMMONEST (Q4, the puzzle)", common)
show("SWEET SPOT (Q3, most stable)", sweet)
show("RAREST (Q1, unstable junk)", rare)
