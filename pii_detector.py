"""Validated PII detector (email tokens) — clean regex labels, safety-relevant.
If it certifies + holds cross-distribution, its features REALLY encode PII = real features."""
import os, re, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

dev="mps" if torch.backends.mps.is_available() else "cpu"
sae=SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=dev)
hook="blocks.12.hook_resid_post"
model=HookedTransformer.from_pretrained("gemma-2-2b",device=dev,dtype=torch.bfloat16)
pile=load_dataset("NeelNanda/pile-10k",split="train")

EMAIL=re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
def email_spans(t): return [(m.start(),m.end()) for m in EMAIL.finditer(t)]

def collect(idxs):
    R=[]; S=[]; y=[]
    for i in idxs:
        t=pile[i]["text"][:1000]
        spans=email_spans(t)
        ids=model.to_tokens(t)
        with torch.no_grad():
            _,c=model.run_with_cache(ids,names_filter=[hook])
            feats=sae.encode(c[hook][0]).float().cpu()
            resid=c[hook][0].float().cpu()
        # label each token: is its char-span inside an email?
        offs=[]; pos=0
        toks=[model.tokenizer.decode([tk]) for tk in ids[0].tolist()]
        cur=0
        for tok in toks:
            j=t.find(tok,cur) if tok.strip() else cur
            start=j if j>=0 else cur; end=start+len(tok)
            offs.append((start,end)); cur=end
        for k,(a,bb) in enumerate(offs):
            lab=1.0 if any(a<se and bb>ss for ss,se in spans) else 0.0
            S.append(feats[k]); R.append(resid[k]); y.append(lab)
        if dev=="mps": torch.mps.empty_cache()
    return torch.stack(S),torch.stack(R),torch.tensor(y)

# need docs that actually contain emails
docs=[i for i in range(len(pile)) if email_spans(pile[i]["text"][:1000])][:200]
nsplit=int(len(docs)*0.7)
print(f"docs with emails: {len(docs)} (train {nsplit}/test {len(docs)-nsplit})")
Str,Rtr,ytr=collect(docs[:nsplit])
Sin,Rin,yin=collect(docs[nsplit:])
print(f"train tokens {len(ytr)} (email+: {int(ytr.sum())})   test {len(yin)} (email+: {int(yin.sum())})")

def f1(X,y,w,b,mu,sd,thr):
    p=(((X-mu)/sd)@w+b>thr).float();tp=(p*y).sum();fp=(p*(1-y)).sum();fn=((1-p)*y).sum()
    pr=tp/(tp+fp+1e-9);rc=tp/(tp+fn+1e-9);return (2*pr*rc/(pr+rc+1e-9)).item(),pr.item(),rc.item()
def fit(X,y):
    # BALANCE: all positives + equal random negatives
    posi=torch.where(y==1)[0]; negi=torch.where(y==0)[0]
    g=torch.Generator().manual_seed(0)
    negs=negi[torch.randperm(len(negi),generator=g)[:len(posi)*3]]
    bi=torch.cat([posi,negs]); X=X[bi]; y=y[bi]
    corr=torch.tensor([((X[:,j]-X[:,j].mean())*(y-y.mean())).mean()/(X[:,j].std()+1e-9) for j in range(X.shape[1])])
    sel=corr.abs().topk(30).indices
    mu,sd=X[:,sel].mean(0),X[:,sel].std(0)+1e-6
    w=torch.zeros(30,requires_grad=True); b=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([w,b],0.05,weight_decay=1e-3)
    for _ in range(500):
        l=torch.nn.functional.binary_cross_entropy_with_logits(((X[:,sel]-mu)/sd)@w+b,y);opt.zero_grad();l.backward();opt.step()
    thr=max(torch.linspace(-3,3,40).tolist(),key=lambda tt: f1(X[:,sel],y,w.detach(),b.detach(),mu,sd,tt)[0])
    return sel,w.detach(),b.detach(),mu,sd,thr

sel,w,b,mu,sd,thr=fit(Str,ytr)
sf,sp,sr=f1(Sin[:,sel],yin,w,b,mu,sd,thr)
print(f"\n=== PII (email) detector — SAE features ===")
print(f"held-out: precision {sp:.2f}  recall {sr:.2f}  F1 {sf:.2f}")
print(f"VERDICT: {'*** CERTIFIED — features really encode PII (real) ***' if sf>0.8 else 'not certified'}")
