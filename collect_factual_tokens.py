"""Literature-driven refinement (Orgad et al.: truthfulness concentrates at the EXACT-ANSWER token,
not mean-pooled). Re-collect factual T/F with residual at SPECIFIC token positions per layer:
 - 'last'  = last token of the claim (Azaria-Mitchell style)
 - 'final' = the position right before the model emits True/False (where it commits)
Tests whether the factual latent-knowledge negative was a pooling artifact."""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from datasets import load_dataset
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
tid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item()
TRUE,FALSE=tid(" True"),tid(" False")
LAYERS=[6,9,12,15,18]
hooks=[f"blocks.{L}.hook_resid_post" for L in LAYERS]
PREFIX=("Statement: The capital of Italy is Rome.\nAnswer: True\n"
        "Statement: The sun rises in the west.\nAnswer: False\n"
        "Statement: Water is made of hydrogen and oxygen.\nAnswer: True\n"
        "Statement: A triangle has four sides.\nAnswer: False\n")

def process(claim):
    pre_claim=PREFIX+f"Statement:{claim}"; full=pre_claim+"\nAnswer:"
    n_end=model.to_tokens(pre_claim).shape[1]; toks=model.to_tokens(full)
    with torch.no_grad():
        logits,cache=model.run_with_cache(toks,names_filter=hooks)
    gap=(logits[0,-1,TRUE]-logits[0,-1,FALSE]).item()
    feats={}
    for L in LAYERS:
        r=cache[f"blocks.{L}.hook_resid_post"][0]
        feats[f"L{L}_last"]=r[n_end-1].float().half().cpu()   # last claim token
        feats[f"L{L}_final"]=r[-1].float().half().cpu()        # answer-commit position
    if dev=="mps": torch.mps.empty_cache()
    return (1 if gap>0 else 0), gap, feats

ds=load_dataset("notrichardren/azaria-mitchell",split="train")
TOPICS=["cities","companies","animals","inventions","elements"]; N=150
out={}
for topic in TOPICS:
    idx=[i for i in range(len(ds)) if ds[i]["dataset"]==topic]
    pos=[i for i in idx if ds[i]["label"]==1][:N//2]; neg=[i for i in idx if ds[i]["label"]==0][:N//2]
    sel=pos+neg
    if not sel: continue
    keys=[f"L{L}_{p}" for L in LAYERS for p in ("last","final")]
    rows={k:[] for k in keys}; J=[];G=[];T=[]
    for i in sel:
        j,g,f=process(ds[i]["claim"])
        for k in keys: rows[k].append(f[k])
        J.append(j);G.append(g);T.append(ds[i]["label"])
    o={k:torch.stack(v) for k,v in rows.items()}; o["judge"]=np.array(J);o["gap"]=np.array(G);o["truth"]=np.array(T)
    out[topic]=o
    print(f"{topic}: n={len(T)} judge_acc={(np.array(J)==np.array(T)).mean():.3f}")
torch.save(out,"/Users/oe/rebuild/factual_tokens.pt")
print("saved factual_tokens.pt")
