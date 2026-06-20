import torch
import torch.nn as nn

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

pile = torch.load("/Users/oe/rebuild/thoughts.pt")

# --- SAE #2: same food, same recipe, different random start ---
torch.manual_seed(123)
sae2 = TinySAE()
opt = torch.optim.Adam(sae2.parameters(), lr=1e-3)
for epoch in range(5):
    perm = torch.randperm(len(pile))
    total = 0; n = 0
    for i in range(0, len(pile), 1024):
        batch = pile[perm[i:i+1024]]
        rebuilt, sparse = sae2(batch)
        loss = ((rebuilt - batch) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item(); n += 1
    print(f"SAE-2 epoch {epoch}: loss {total/n:.4f}")
torch.save(sae2.state_dict(), "/Users/oe/rebuild/sae2.pt")

# --- the matching ---
sae1 = TinySAE()
sae1.load_state_dict(torch.load("/Users/oe/rebuild/sae.pt"))

with torch.no_grad():
    W1 = sae1.W_dec.weight        # (768, 2048), columns = feature arrows
    W2 = sae2.W_dec.weight
    W1n = W1 / W1.norm(dim=0)
    W2n = W2 / W2.norm(dim=0)
    C = W1n.T @ W2n               # (2048, 2048) cosines — 4.2M hand-calcs at once
    best = C.max(dim=1).values

for thresh in [0.9, 0.8, 0.7, 0.5]:
    rate = (best > thresh).float().mean().item()
    print(f"replication rate at cosine>{thresh}: {rate*100:.1f}%")
print("median best-match cosine:", best.median().item())
