"""Frontier collection on HOW the model represents facts: for each factual T/F claim, cache the
RAW residual stream at several layers (pooled over claim tokens) + the model's True/False logit gap
(its confidence) + its judgment + ground truth. Enables: (A) where across layers is truth linearly
decodable? (B) does any layer's internal state sense the model's factual errors beyond its confidence?"""
import os, torch, numpy as np
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from datasets import load_dataset
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
tid=lambda s: model.to_tokens(s,prepend_bos=False)[0,0].item()
TRUE,FALSE=tid(" True"),tid(" False")
LAYERS=[3,6,9,12,15,18,21,24]
hooks=[f"blocks.{L}.hook_resid_post" for L in LAYERS]
PREFIX=("Statement: The capital of Italy is Rome.\nAnswer: True\n"
        "Statement: The sun rises in the west.\nAnswer: False\n"
        "Statement: Water is made of hydrogen and oxygen.\nAnswer: True\n"
        "Statement: A triangle has four sides.\nAnswer: False\n")
n_pre=model.to_tokens(PREFIX+"Statement:").shape[1]

def process(claim):
    pre_claim=PREFIX+f"Statement:{claim}"; full=pre_claim+"\nAnswer:"
    n_end=model.to_tokens(pre_claim).shape[1]; toks=model.to_tokens(full)
    with torch.no_grad():
        logits,cache=model.run_with_cache(toks,names_filter=hooks)
    gap=(logits[0,-1,TRUE]-logits[0,-1,FALSE]).item()
    feats={f"L{L}":cache[f"blocks.{L}.hook_resid_post"][0,n_pre:n_end].float().mean(0).half().cpu() for L in LAYERS}
    if dev=="mps": torch.mps.empty_cache()
    return (1 if gap>0 else 0), gap, feats

ds=load_dataset("notrichardren/azaria-mitchell",split="train")
TOPICS=["cities","companies","animals","inventions","elements"]; N=150
out={}
for topic in TOPICS:
    idx=[i for i in range(len(ds)) if ds[i]["dataset"]==topic]
    pos=[i for i in idx if ds[i]["label"]==1][:N//2]; neg=[i for i in idx if ds[i]["label"]==0][:N//2]
    sel=pos+neg
    if not sel: print("skip",topic); continue
    rows={f"L{L}":[] for L in LAYERS}; J=[];G=[];T=[]
    for i in sel:
        j,g,f=process(ds[i]["claim"])
        for L in LAYERS: rows[f"L{L}"].append(f[f"L{L}"])
        J.append(j);G.append(g);T.append(ds[i]["label"])
    o={k:torch.stack(v) for k,v in rows.items()}; o["judge"]=np.array(J);o["gap"]=np.array(G);o["truth"]=np.array(T)
    out[topic]=o
    print(f"{topic}: n={len(T)} judge_acc={(np.array(J)==np.array(T)).mean():.3f}")
torch.save(out,"/Users/oe/rebuild/factual_layers.pt")
print("saved factual_layers.pt")
