"""Pre-collect Gemma Scope SAE features so an agent fleet can search detector recipes
without each reloading Gemma. 3 distributions: Pile (train/test) + TinyStories + wikitext (OOD)."""
import os, torch, string
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
pile=load_dataset("NeelNanda/pile-10k",split="train")
ts=load_dataset("roneneldan/TinyStories",split="train")
try: wk=load_dataset("wikitext","wikitext-2-raw-v1",split="train"); wkget=lambda i: wk[i*7]["text"]
except Exception: wk=None; wkget=lambda i: pile[i+300]["text"]

def labs(tok):
    t=tok.strip()
    return {"newline":1.0 if "\n" in tok else 0.0,"comma":1.0 if t=="," else 0.0,
            "period":1.0 if t=="." else 0.0,"digit":1.0 if (t.isdigit() and t) else 0.0,
            "space_pre":1.0 if tok.startswith(" ") else 0.0,
            "cap_start":1.0 if (len(t)>0 and t[0].isupper()) else 0.0}
KEYS=list(labs("x"))

def collect(get,idxs,maxtok=9000):
    F=[]; L={k:[] for k in KEYS}; n=0
    for i in idxs:
        t=get(i)
        if not t or len(t)<20: continue
        ids=model.to_tokens(t[:500])
        with torch.no_grad():
            _,c=model.run_with_cache(ids,names_filter=[hook])
            f=sae.encode(c[hook][0]).half().cpu()      # float16 to save space
        F.append(f)
        for p in ids[0].tolist():
            d=labs(model.tokenizer.decode([p]))
            for k in KEYS: L[k].append(d[k])
        n+=f.shape[0]
        if dev=="mps": torch.mps.empty_cache()
        if n>=maxtok: break
    return torch.cat(F),{k:torch.tensor(v) for k,v in L.items()}

print("collecting Gemma features: Pile train/test, TinyStories, wikitext...")
trF,trL=collect(lambda i: pile[i]["text"], range(0,120))
teF,teL=collect(lambda i: pile[i]["text"], range(120,180))
tsF,tsL=collect(lambda i: ts[i]["text"], range(0,120))
wkF,wkL=collect(wkget, range(0,120))
torch.save({"keys":KEYS,"train_F":trF,"train_L":trL,"test_F":teF,"test_L":teL,
            "ts_F":tsF,"ts_L":tsL,"wk_F":wkF,"wk_L":wkL},
           "/Users/oe/rebuild/gemma_detector_dataset.pt")
for nm,F,L in [("train",trF,trL),("test",teF,teL),("TinyStories",tsF,tsL),("wikitext",wkF,wkL)]:
    pos={k:int(L[k].sum()) for k in KEYS}
    print(f"{nm}: {F.shape[0]} tokens  positives={pos}")
print("saved /Users/oe/rebuild/gemma_detector_dataset.pt")
