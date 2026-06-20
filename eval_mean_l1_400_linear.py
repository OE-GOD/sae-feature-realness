import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "mean"; nf = 400

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

# l1 selection
sel = LogisticRegression(penalty="l1", solver="liblinear", C=0.5)
sel.fit(Xtr, ytr)
coef = np.abs(sel.coef_.ravel())
cols = np.argsort(coef)[::-1][:nf]

probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr[:, cols], ytr)

def f1(X, y): return f1_score(y, probe.predict(X[:, cols]))
indist = f1(Xte, yte); rt = f1(Xrt, yrt); am = f1(Xam, yam)
mean_ood = (rt + am) / 2
print(f"{indist:.6f} {rt:.6f} {am:.6f} {mean_ood:.6f}")
