"""Industrial validation = detectors must hold ACROSS DISTRIBUTIONS, not just
held-out same-distribution docs. Build on Pile, test on Pile (in-dist) AND a
different corpus (out-of-dist). Industrial-certified = passes BOTH."""
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
# out-of-distribution corpus (different from Pile)
try:
    ood=load_dataset("roneneldan/TinyStories",split="train"); ood_name="TinyStories"
    ood_txt=lambda i: ood[i]["text"]
except Exception as e:
    ood=load_dataset("wikitext","wikitext-2-raw-v1",split="train"); ood_name="wikitext"
    ood_txt=lambda i: ood[i]["text"]
print(f"OOD corpus: {ood_name}")

def labels(tok):
    t=tok.strip()
    return {"newline":1.0 if "\n" in tok else 0.0,
            "digit":1.0 if (t.isdigit() and t) else 0.0,
            "punct":1.0 if t in string.punctuation else 0.0,
            "cap_word":1.0 if (len(t)>1 and t[0].isupper() and t[1:].islower()) else 0.0,
            "space_pre":1.0 if tok.startswith(" ") else 0.0}
KEYS=list(labels("x"))

def collect(get_text, idxs):
    acts=[]; labs={k:[] for k in KEYS}
    for i in idxs:
        txt=get_text(i)
        if not txt or len(txt)<20: continue
        ids=model.to_tokens(txt[:500])
        with torch.no_grad():
            _,c=model.run_with_cache(ids,names_filter=[hook])
            f=sae.encode(c[hook][0]).float().cpu()
        acts.append(f)
        for p in ids[0].tolist():
            L=labels(model.tokenizer.decode([p]))
            for k in KEYS: labs[k].append(L[k])
        if dev=="mps": torch.mps.empty_cache()
    return torch.cat(acts), {k:torch.tensor(v) for k,v in labs.items()}

Xtr,Ytr=collect(lambda i: pile[i]["text"], range(0,60))
Xin,Yin=collect(lambda i: pile[i]["text"], range(60,90))      # in-distribution test
Xood,Yood=collect(ood_txt, range(0,60))                        # out-of-distribution test
print(f"train {len(next(iter(Ytr.values())))}  in-dist {len(next(iter(Yin.values())))}  OOD {len(next(iter(Yood.values())))}\n")

def f1(Z,y,w,b,thr):
    pred=((Z@w+b)>thr).float(); tp=(pred*y).sum(); fp=(pred*(1-y)).sum(); fn=((1-pred)*y).sum()
    p=tp/(tp+fp+1e-9); r=tp/(tp+fn+1e-9); return (2*p*r/(p+r+1e-9)).item()

print(f"{'CONCEPT':10} {'in-dist F1':>10} {'OOD F1':>8}  INDUSTRIAL VERDICT")
ncert=0
for k in KEYS:
    ytr=Ytr[k]
    if ytr.sum()<5: print(f"{k:10} {'(too rare in train)':>20}"); continue
    corr=torch.tensor([((Xtr[:,j]-Xtr[:,j].mean())*(ytr-ytr.mean())).mean()/(Xtr[:,j].std()+1e-9) for j in range(Xtr.shape[1])])
    sel=corr.abs().topk(30).indices
    mu,sd=Xtr[:,sel].mean(0),Xtr[:,sel].std(0)+1e-6
    Ztr=(Xtr[:,sel]-mu)/sd
    w=torch.zeros(30,requires_grad=True); b=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([w,b],0.05)
    for _ in range(400):
        loss=torch.nn.functional.binary_cross_entropy_with_logits(Ztr@w+b,ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    w=w.detach(); b=b.detach()
    thr=max(torch.linspace(-3,3,40).tolist(), key=lambda t: f1(Ztr,ytr,w,b,t))
    Zin=(Xin[:,sel]-mu)/sd; Zood=(Xood[:,sel]-mu)/sd
    fin=f1(Zin,Yin[k],w,b,thr); food=f1(Zood,Yood[k],w,b,thr)
    ok=fin>0.8 and food>0.7
    if ok: ncert+=1
    print(f"{k:10} {fin:>10.2f} {food:>8.2f}  {'*** INDUSTRIAL-CERTIFIED (holds cross-distribution) ***' if ok else 'fails cross-distribution' if fin>0.8 else 'not reliable'}")
print(f"\n{ncert}/{len(KEYS)} detectors INDUSTRIAL-CERTIFIED (in-dist F1>0.8 AND OOD F1>0.7)")
