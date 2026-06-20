"""Pre-collect a shared feature dataset so a fleet of agents can search detector recipes
without each reloading the model. Pythia toy SAE; concepts x {Pile train/test, TinyStories OOD}."""
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
model=HookedTransformer.from_pretrained("pythia-160m")
pile=load_dataset("NeelNanda/pile-10k",split="train")
ts=load_dataset("roneneldan/TinyStories",split="train")

def labs(tok):
    t=tok.strip()
    return {"newline":1.0 if "\n" in tok else 0.0,"comma":1.0 if t=="," else 0.0,
            "period":1.0 if t=="." else 0.0,"digit":1.0 if (t.isdigit() and t) else 0.0,
            "space_pre":1.0 if tok.startswith(" ") else 0.0,
            "cap_start":1.0 if (len(t)>0 and t[0].isupper()) else 0.0}
KEYS=list(labs("x"))

def collect(get,idxs):
    F=[]; L={k:[] for k in KEYS}
    for i in idxs:
        t=get(i)
        if not t or len(t)<20: continue
        ids=model.to_tokens(t[:500])
        with torch.no_grad():
            _,c=model.run_with_cache(ids,names_filter=["blocks.6.hook_resid_post"])
            _,sp=sae(c["blocks.6.hook_resid_post"][0])
        F.append(sp.cpu())
        for p in ids[0].tolist():
            d=labs(model.tokenizer.decode([p]))
            for k in KEYS: L[k].append(d[k])
    return torch.cat(F), {k:torch.tensor(v) for k,v in L.items()}

print("collecting Pile train/test + TinyStories OOD...")
ds={}
ds["train"]=collect(lambda i: pile[i]["text"], range(0,70))
ds["test"] =collect(lambda i: pile[i]["text"], range(70,100))
ds["ood"]  =collect(lambda i: ts[i]["text"],  range(0,50))
torch.save({"keys":KEYS,
            "train_F":ds["train"][0],"train_L":ds["train"][1],
            "test_F":ds["test"][0],"test_L":ds["test"][1],
            "ood_F":ds["ood"][0],"ood_L":ds["ood"][1]},
           "/Users/oe/rebuild/detector_dataset.pt")
for s in ["train","test","ood"]:
    print(f"{s}: {ds[s][0].shape[0]} tokens")
print("saved /Users/oe/rebuild/detector_dataset.pt  (concepts:",KEYS,")")
