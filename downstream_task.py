"""Industrial certification via DOWNSTREAM TASK (sidesteps the no-oracle problem:
the task has real labels). A feature is 'real for use X' if firing reliably predicts
label X on HELD-OUT data AND is stable across seeds."""
import torch, torch.nn as nn, string
from transformer_lens import HookedTransformer
from datasets import load_dataset

class TinySAE(nn.Module):
    def __init__(s,d=768,n=2048,k=32):
        super().__init__(); s.k=k; s.W_enc=nn.Linear(d,n); s.W_dec=nn.Linear(n,d)
    def forward(s,x):
        sc=s.W_enc(x); tk=torch.topk(sc,s.k,-1)
        sp=torch.zeros_like(sc); sp.scatter_(-1,tk.indices,tk.values)
        return s.W_dec(sp),sp

sae=TinySAE(); sae.load_state_dict(torch.load("/Users/oe/rebuild/sae.pt")); sae.eval()
sae2=TinySAE(); sae2.load_state_dict(torch.load("/Users/oe/rebuild/sae2.pt")); sae2.eval()
model=HookedTransformer.from_pretrained("pythia-160m")
data=load_dataset("NeelNanda/pile-10k",split="train")
N=2048
with torch.no_grad():
    W1=sae.W_dec.weight; W2=sae2.W_dec.weight
    STAB=((W1/W1.norm(dim=0)).T@(W2/W2.norm(dim=0))).max(1).values

# tasks: per-token labels from the token string
def lab(tok):
    t=tok.strip()
    return {"digit":t.isdigit() and t!="", "newline":"\n" in tok,
            "punct":t in string.punctuation, "cap_word":len(t)>1 and t[0].isupper() and t[1:].islower(),
            "space_pre":tok.startswith(" ")}

acts=[]; labels={k:[] for k in lab("x")}
for i in range(120):
    ids=model.to_tokens(data[i]["text"][:600])
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=["blocks.6.hook_resid_post"])
        _,sp=sae(c["blocks.6.hook_resid_post"][0])
    acts.append((sp>0).float())
    for p in ids[0].tolist():
        L=lab(model.tokenizer.decode([p]))
        for k in labels: labels[k].append(1.0 if L[k] else 0.0)
F=torch.cat(acts)                      # [T, N] binary firing
T=F.shape[0]; ntr=int(T*0.7)
print(f"tokens {T}, train {ntr} / test {T-ntr}\n")

def f1(pred,lab):
    tp=(pred*lab).sum(); fp=(pred*(1-lab)).sum(); fn=((1-pred)*lab).sum()
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return (2*p*r/(p+r+1e-9)).item()

print(f"{'TASK':10} {'best_f':>7} {'trainF1':>8} {'HELD-OUT F1':>12} {'stability':>10}  VERDICT")
for k in labels:
    y=torch.tensor(labels[k])
    ytr,yte=y[:ntr],y[ntr:]; Ftr,Fte=F[:ntr],F[ntr:]
    # pick best feature on TRAIN by F1
    f1s=torch.tensor([f1(Ftr[:,j],ytr) for j in range(N)])
    bf=f1s.argmax().item()
    held=f1(Fte[:,bf],yte)             # HELD-OUT performance
    st=STAB[bf].item()
    ok = held>0.8 and st>0.7
    print(f"{k:10} {bf:>7} {f1s[bf]:>8.2f} {held:>12.2f} {st:>10.2f}  {'*** CERTIFIED for use ***' if ok else 'not reliable'}")
