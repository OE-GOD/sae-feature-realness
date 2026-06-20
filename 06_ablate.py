import torch
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

sae = TinySAE()
sae.load_state_dict(torch.load("/Users/oe/rebuild/sae.pt"))
sae.eval()

model = HookedTransformer.from_pretrained("pythia-160m")
prompt = "When Mary and John went to the store, John gave a drink to"
tokens = model.to_tokens(prompt)

logits, cache = model.run_with_cache(tokens)
resid = cache["blocks.6.hook_resid_post"]          # (1, seq, 768)
last_thought = resid[0, -1].cpu()

with torch.no_grad():
    _, sparse = sae(last_thought)

vals, idx = sparse.topk(5)
print("top firing features on final token ' to':")
for v, f in zip(vals, idx):
    print(f"   feature {f.item():5d}  activation {v.item():7.2f}")

mary = model.to_single_token(" Mary")
john = model.to_single_token(" John")
base_probs = logits[0, -1].softmax(-1)
print(f"\nBASE:  P(' Mary') = {base_probs[mary].item():.4f}   P(' John') = {base_probs[john].item():.4f}")

def ablate(f):
    act = sparse[f].item()
    col = sae.W_dec.weight[:, f].detach()
    def hook(resid_post, hook):
        resid_post[0, -1] = resid_post[0, -1] - act * col.to(resid_post.device)
        return resid_post
    abl = model.run_with_hooks(tokens, fwd_hooks=[("blocks.6.hook_resid_post", hook)])
    p = abl[0, -1].softmax(-1)
    return p[mary].item(), p[john].item()

target = idx[0].item()
control = idx[4].item()

m, j = ablate(target)
print(f"\nABLATE feature {target} (TOP firing, act {vals[0].item():.1f}):")
print(f"       P(' Mary') = {m:.4f}   P(' John') = {j:.4f}")

m2, j2 = ablate(control)
print(f"\nABLATE feature {control} (CONTROL, 5th firing, act {vals[4].item():.1f}):")
print(f"       P(' Mary') = {m2:.4f}   P(' John') = {j2:.4f}")
