import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
emb = cache["hook_embed"][0, -1].cpu()
x6  = cache["blocks.6.hook_resid_post"][0, -1].cpu()
with torch.no_grad():
    xhat, sparse = sae(x6)
cos = torch.nn.functional.cosine_similarity(x6, xhat, dim=0).item()

# ---------- FIG 1: the pipeline ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, vec, title in [
    (axes[0], emb, "STATION 1: word as raw numbers\n(embedding — just 'today', no understanding)"),
    (axes[1], x6, "STATION 2/3: THE GOO after 7 layers\n(interview+negation+bad-vibe smeared in — unreadable)"),
    (axes[2], xhat, f"REBUILT by your SAE from 32 ingredients\n(cosine {cos:.3f} to the goo)")]:
    im = ax.imshow(vec.reshape(24, 32), cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Your sentence: "The interview did not go well today" — last token, 768 numbers as 24x32 grids', fontsize=12)
fig.colorbar(im, ax=axes, shrink=0.7, label="value (red +, blue -)")
plt.savefig("/Users/oe/rebuild/fig1_pipeline.png", dpi=150, bbox_inches="tight")
print("fig1 saved")

# ---------- FIG 2: the recipe (sparsity made visible) ----------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7))
a1.bar(range(2048), sparse.numpy(), width=1.5, color="#1f6feb")
a1.set_title("THE RECIPE: all 2,048 ingredient scores for this thought — the bouncer left only 32 alive (the spikes)", fontsize=11)
a1.set_xlabel("feature #"); a1.set_ylabel("activation")
vals, idx = sparse.topk(10)
names = [f"f{int(f)}" for f in idx]
a2.barh(names[::-1], vals.numpy()[::-1], color="#58a6ff")
a2.set_title("top 10 ingredients (f250 & f1467 = your busiest features; f1853 = the thermometer you ablated)", fontsize=11)
a2.set_xlabel("activation strength")
plt.tight_layout()
plt.savefig("/Users/oe/rebuild/fig2_recipe.png", dpi=150, bbox_inches="tight")
print("fig2 saved")

# ---------- FIG 3: the replication histogram ----------
sae2 = TinySAE(); sae2.load_state_dict(torch.load("/Users/oe/rebuild/sae2.pt")); sae2.eval()
with torch.no_grad():
    W1 = sae.W_dec.weight;  W1n = W1 / W1.norm(dim=0)
    W2 = sae2.W_dec.weight; W2n = W2 / W2.norm(dim=0)
    best = (W1n.T @ W2n).max(dim=1).values
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(best.numpy(), bins=80, color="#1f6feb", edgecolor="none")
ax.axvline(0.9, color="#f85149", linestyle="--", linewidth=2)
ax.text(0.905, ax.get_ylim()[1]*0.85, "cosine 0.9 threshold\nonly 1.0% live here →", color="#f85149", fontsize=11)
ax.set_title("YOUR REPLICATION EXPERIMENT: each of SAE-1's 2,048 features vs its best twin in SAE-2\n(same food, same recipe, different seed — identical loss 0.1056, almost no shared features)", fontsize=11)
ax.set_xlabel("best-match cosine (1.0 = perfect twin)"); ax.set_ylabel("# features")
plt.tight_layout()
plt.savefig("/Users/oe/rebuild/fig3_replication.png", dpi=150, bbox_inches="tight")
print("fig3 saved")
