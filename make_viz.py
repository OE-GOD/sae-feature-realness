import torch, json
import torch.nn as nn
from transformer_lens import HookedTransformer

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
        return self.W_dec(sparse), sparse

sae = TinySAE(); sae.load_state_dict(torch.load("/Users/oe/rebuild/sae.pt")); sae.eval()
model = HookedTransformer.from_pretrained("pythia-160m")

text = "The interview did not go well today"
logits, cache = model.run_with_cache(text)
toks = model.to_str_tokens(text)

# embedding (station 1) vs layer-6 thought (station 2/3) for the LAST token
emb = cache["hook_embed"][0, -1].cpu()
x6  = cache["blocks.6.hook_resid_post"][0, -1].cpu()
with torch.no_grad():
    xhat, sparse = sae(x6)

cos = torch.nn.functional.cosine_similarity(x6, xhat, dim=0).item()
vals, idx = sparse.topk(10)
top = [{"f": int(f), "v": round(float(v), 2)} for v, f in zip(vals, idx) if v > 0]

data = {
    "tokens": toks, "cos": round(cos, 3), "top": top,
    "emb": [round(float(v), 3) for v in emb],
    "x6": [round(float(v), 3) for v in x6],
    "xhat": [round(float(v), 3) for v in xhat],
}
json.dump(data, open("/Users/oe/rebuild/viz_data.json", "w"))
print("data exported. cos(x6, xhat) =", round(cos, 3), "| top features:", top[:5])
