import os, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS","1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

device = "mps" if torch.backends.mps.is_available() else "cpu"
sae = SAE.from_pretrained("gemma-scope-2b-pt-res","layer_12/width_16k/average_l0_82",device=device)
hook = sae.cfg.metadata["hook_name"]
print(f"REAL SAE: d_sae={sae.cfg.d_sae}, hook={hook}")
model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=torch.bfloat16)
data = load_dataset("NeelNanda/pile-10k", split="train")

D = sae.cfg.d_sae
freq = torch.zeros(D)
allacts=[]; CTX=[]; ntok=0
for i in range(30):
    ids = model.to_tokens(data[i]["text"][:500])
    toks = [model.tokenizer.decode([t]) for t in ids[0].tolist()]
    with torch.no_grad():
        _,c = model.run_with_cache(ids, names_filter=[hook])
        f = sae.encode(c[hook][0]).float().cpu()   # [seq, D]
    freq += (f>0).float().sum(0)
    allacts.append(f); ntok += f.shape[0]
    for p,t in enumerate(toks):
        CTX.append(("".join(toks[max(0,p-4):p])+"«"+t+"»"+"".join(toks[p+1:p+3])).replace("\n","/"))
    if device=="mps": torch.mps.empty_cache()
freq/=ntok
A=torch.cat(allacts)   # [ntok, D]
print(f"tokens: {ntok}")

fires=(A>0).float().sum(0)
cand=torch.where(fires>=5)[0]
order=cand[freq[cand].argsort()]
rare=order[0].item(); mid=order[len(order)//2].item(); common=order[-1].item()

def show(label,fidx):
    print(f"\n=== {label}: feature {fidx} (fires on {freq[fidx]*100:.2f}% of tokens) ===")
    v,idx=A[:,fidx].topk(8)
    for vv,j in zip(v,idx):
        if vv.item()==0: break
        print(f"   {vv.item():6.2f}  {CTX[j][:80]}")

show("RAREST", rare); show("MIDDLE", mid); show("COMMONEST", common)
