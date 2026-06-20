"""Scale the cross-distribution finding: does in-distribution F1 predict OOD F1?
8 concepts x 3 distributions. If they diverge, in-dist evals (SAEBench) can't certify real features."""
import os, string, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook=sae.cfg.metadata["hook_name"]
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)

def labels(tok):
    t=tok.strip()
    return {"newline":"\n" in tok,"digit":t.isdigit() and t!="","space_pre":tok.startswith(" "),
            "punct":t in string.punctuation,"comma":t==",","period":t==".",
            "cap_start":len(t)>0 and t[0].isupper(),"upper":len(t)>1 and t.isupper()}
KEYS=list(labels("x"))

def collect(get_text,n):
    F=[]; L={k:[] for k in KEYS}
    for i in range(n):
        t=get_text(i)
        if not t or len(t)<20: continue
        ids=model.to_tokens(t[:500])
        with torch.no_grad():
            _,c=model.run_with_cache(ids,names_filter=[hook])
            f=(sae.encode(c[hook][0])>0).float().cpu()
        F.append(f)
        for p in ids[0].tolist():
            d=labels(model.tokenizer.decode([p]))
            for k in KEYS: L[k].append(1.0 if d[k] else 0.0)
        if dev=="mps": torch.mps.empty_cache()
    return torch.cat(F),{k:torch.tensor(v) for k,v in L.items()}

pile=load_dataset("NeelNanda/pile-10k",split="train")
ts=load_dataset("roneneldan/TinyStories",split="train")
try: wk=load_dataset("wikitext","wikitext-2-raw-v1",split="train"); wkget=lambda i: wk[i*5]["text"]
except Exception: wk=None; wkget=lambda i: pile[i+200]["text"]
print("collecting Pile(train+test)/TinyStories/wikitext...")
Ftr,Ltr=collect(lambda i: pile[i]["text"],50)
Fte,Lte=collect(lambda i: pile[50+i]["text"],30)
Fts,Lts=collect(lambda i: ts[i]["text"],40)
Fwk,Lwk=collect(wkget,40)

def f1(F,y,w,b,mu,sd,thr):
    p=(((F[:,sel]-mu)/sd)@w+b>thr).float();tp=(p*y).sum();fp=(p*(1-y)).sum();fn=((1-p)*y).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return (2*pr*rc/(pr+rc+1e-9)).item()

print(f"\n{'concept':10} {'in-dist':>8} {'TinySt':>8} {'wikitext':>9}")
rows=[]
for k in KEYS:
    ytr=Ltr[k]
    if ytr.sum()<5: continue
    corr=torch.tensor([((Ftr[:,j]-Ftr[:,j].mean())*(ytr-ytr.mean())).mean()/(Ftr[:,j].std()+1e-9) for j in range(Ftr.shape[1])])
    sel=corr.abs().topk(30).indices
    mu,sd=Ftr[:,sel].mean(0),Ftr[:,sel].std(0)+1e-6
    w=torch.zeros(30,requires_grad=True); b=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([w,b],0.05,weight_decay=1e-3)
    for _ in range(400):
        l=torch.nn.functional.binary_cross_entropy_with_logits(((Ftr[:,sel]-mu)/sd)@w+b,ytr);opt.zero_grad();l.backward();opt.step()
    w,b=w.detach(),b.detach()
    thr=max(torch.linspace(-3,3,30).tolist(),key=lambda tt:f1(Ftr,ytr,w,b,mu,sd,tt))
    ind=f1(Fte,Lte[k],w,b,mu,sd,thr)
    fts=f1(Fts,Lts[k],w,b,mu,sd,thr) if Lts[k].sum()>=3 else float('nan')
    fwk=f1(Fwk,Lwk[k],w,b,mu,sd,thr) if Lwk[k].sum()>=3 else float('nan')
    rows.append((k,ind,fts,fwk))
    print(f"{k:10} {ind:>8.2f} {fts:>8.2f} {fwk:>9.2f}")

# key analysis: does in-dist predict OOD?
import statistics
pairs=[(ind,o) for _,ind,a,b2 in rows for o in (a,b2) if o==o]
xs=[p[0] for p in pairs]; ys=[p[1] for p in pairs]
mx,my=statistics.mean(xs),statistics.mean(ys)
cov=sum((a-mx)*(b2-my) for a,b2 in pairs); sx=sum((a-mx)**2 for a in xs)**.5; sy=sum((b2-my)**2 for b2 in ys)**.5
r=cov/(sx*sy) if sx*sy else 0
cert_in=[(k,ind,a,b2) for k,ind,a,b2 in rows if ind>0.8]
failed_ood=[k for k,ind,a,b2 in cert_in if (a==a and a<0.7) or (b2==b2 and b2<0.7)]
print(f"\nin-dist certified (F1>0.8): {len(cert_in)}/{len(rows)}")
print(f"...of those, FAILED cross-distribution (OOD<0.7): {len(failed_ood)} -> {failed_ood}")
print(f"Pearson(in-dist F1, OOD F1) across all pairs = {r:.2f}")
print(f"interpretation: low r / many OOD failures => in-dist eval CANNOT certify real features")
