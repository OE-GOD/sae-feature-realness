"""Pre-compute sentiment SAE features under 4 POOLING strategies, 3 distributions,
so a fleet can search what pooling fixes the semantic cross-distribution gap."""
import os, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)

def pools(text):
    ids=model.to_tokens(text[:300])
    with torch.no_grad():
        _,c=model.run_with_cache(ids,names_filter=[hook])
        f=sae.encode(c[hook][0]).float()        # [seq,16384]
    return {"mean":f.mean(0).half().cpu(),"last":f[-1].half().cpu(),
            "max":f.max(0).values.half().cpu(),"meanlast3":f[-3:].mean(0).half().cpu()}

def collect(texts):
    out={k:[] for k in ["mean","last","max","meanlast3"]}
    for t in texts:
        if not t or len(t)<5: 
            for k in out: out[k].append(torch.zeros(16384,dtype=torch.half))
            continue
        p=pools(t)
        for k in out: out[k].append(p[k])
        if dev=="mps": torch.mps.empty_cache()
    return {k:torch.stack(v) for k,v in out.items()}

def balanced(ds,txt,lab,npos,off=0):
    P=[i for i in range(off,len(ds)) if lab(i)==1][:npos]; N=[i for i in range(off,len(ds)) if lab(i)==0][:npos]
    return [txt(i) for i in P+N], torch.tensor([1]*len(P)+[0]*len(N))

print("loading sentiment datasets...")
sst=load_dataset("stanfordnlp/sst2",split="train")
rt=load_dataset("rotten_tomatoes",split="train")
try: am=load_dataset("amazon_polarity",split="train"); amtx=lambda i:am[i]["content"]; amlb=lambda i:am[i]["label"]; amn="amazon_polarity"
except Exception:
    am=load_dataset("imdb",split="train"); amtx=lambda i:am[i]["text"]; amlb=lambda i:am[i]["label"]; amn="imdb"
print("OOD-2:",amn)

trX,try_=balanced(sst,lambda i:sst[i]["sentence"],lambda i:sst[i]["label"],300)
teX,tey =balanced(sst,lambda i:sst[i]["sentence"],lambda i:sst[i]["label"],100,off=600)
rtX,rty =balanced(rt,lambda i:rt[i]["text"],lambda i:rt[i]["label"],150)
amX,amy =balanced(am,amtx,amlb,150)
print(f"train {len(try_)} / test {len(tey)} / rotten {len(rty)} / {amn} {len(amy)}  -- encoding (slow)...")
data={"train":(collect(trX),try_),"test":(collect(teX),tey),
      "rt":(collect(rtX),rty),"am":(collect(amX),amy),"am_name":amn}
torch.save(data,"/Users/oe/rebuild/sem_pooling_dataset.pt")
print("saved /Users/oe/rebuild/sem_pooling_dataset.pt  (poolings: mean,last,max,meanlast3)")
