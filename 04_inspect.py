import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from datasets import load_dataset

class TinySAE(nn.Module):
    def __init__(self, d_model=768, n_features=2048, k=32):
        super().__init__()
        self.k = k
        self.W_enc = nn.Linear(d_model, n_features)
        self.W_dec = nn.Linear(n_features, d_model)
    def forward(self, x):
        scores = self.W_enc(x)
        topk = torch.topk(scores, self.k, dim=-1)
        sparse = torch.zeros_like(scores)
        sparse.scatter_(-1, topk.indices, topk.values)
        rebuilt = self.W_dec(sparse)
        return rebuilt, sparse

sae = TinySAE()
sae.load_state_dict(torch.load("/Users/oe/rebuild/sae.pt"))
sae.eval()

model = HookedTransformer.from_pretrained("pythia-160m")
data = load_dataset("NeelNanda/pile-10k", split="train")

all_sparse = []
all_ctx = []
for i in range(50):
    text = data[i]["text"][:2000]
    with torch.no_grad():
        logits, cache = model.run_with_cache(text)
    thoughts = cache["blocks.6.hook_resid_post"][0].cpu()
    with torch.no_grad():
        rebuilt, sparse = sae(thoughts)
    toks = model.to_str_tokens(text)
    all_sparse.append(sparse)
    for p, t in enumerate(toks):
        ctx = "".join(toks[max(0,p-5):p]) + "«" + t + "»" + "".join(toks[p+1:min(len(toks),p+4)])
        all_ctx.append(ctx.replace("\n", "↵"))

S = torch.cat(all_sparse)
print("sparse codes for", S.shape[0], "tokens, 2048 features each")

def show(f, label):
    vals, idx = S[:, f].topk(8)
    print(f"\n=== FEATURE {f} ({label}) ===")
    if vals[0].item() == 0:
        print("   DEAD — no token in 50 docs ever fired it")
        return
    for v, j in zip(vals, idx):
        if v.item() == 0: break
        print(f"  {v.item():7.2f}   {all_ctx[j][:90]}")

busiest = (S > 0).float().sum(0).topk(3).indices.tolist()
print("\nBUSIEST FEATURES:", busiest)
for f in busiest: show(f, "busiest")
for f in [87, 73, 989]: show(f, "your pick")
