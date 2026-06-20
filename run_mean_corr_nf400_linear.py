import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "mean"; nf = 400

def get(split):
    P, y = data[split]
    return P[pooling].float().numpy(), y.numpy()

Xtr, ytr = get("train")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

# corr selection: |point-biserial corr| with ytr
yc = (ytr - ytr.mean())
corr = (Xtr_s * yc[:, None]).mean(0) / (Xtr_s.std(0) * yc.std() + 1e-12)
cols = np.argsort(-np.abs(corr))[:nf]

probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_s[:, cols], ytr)

def evalf(split):
    X, y = get(split)
    Xs = ((X - mu) / sd)[:, cols]
    return f1_score(y, probe.predict(Xs))

indist = evalf("test"); rt = evalf("rt"); am = evalf("am")
mean_ood = (rt + am) / 2
print(f"indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={mean_ood:.4f}")
