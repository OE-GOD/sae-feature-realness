import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling, nf = "mean", 100

def get(split):
    P, y = data[split]
    return P[pooling].float().numpy(), y.numpy()

Xtr, ytr = get("train")
Xte, yte = get("test")
Xrt, yrt = get("rt")
Xam, yam = get("am")

mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
def std(X): return (X - mu) / sd
Xtr, Xte, Xrt, Xam = std(Xtr), std(Xte), std(Xrt), std(Xam)

# corr selection: point-biserial = pearson corr with binary label
yc = ytr - ytr.mean()
denom = (Xtr.std(0) * yc.std())
denom[denom == 0] = 1.0
corr = (Xtr * yc[:, None]).mean(0) / denom
cols = np.argsort(-np.abs(corr))[:nf]

probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr[:, cols], ytr)

def f1(X, y): return f1_score(y, probe.predict(X[:, cols]))
indist = f1(Xte, yte); rt = f1(Xrt, yrt); am = f1(Xam, yam)
print(f"indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={(rt+am)/2:.4f}")
