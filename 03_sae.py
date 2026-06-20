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

if __name__ == "__main__":
    pile = torch.load("/Users/oe/rebuild/thoughts.pt")
    sae = TinySAE()
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)

    for epoch in range(5):
        perm = torch.randperm(len(pile))
        total = 0
        n_batches = 0
        for i in range(0, len(pile), 1024):
            batch = pile[perm[i:i+1024]]
            rebuilt, sparse = sae(batch)
            loss = ((rebuilt - batch) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n_batches += 1
        print(f"epoch {epoch}: loss {total / n_batches:.4f}")

    torch.save(sae.state_dict(), "/Users/oe/rebuild/sae.pt")
    print("SAE saved. The translator exists.")
