import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling, nf = "max", 100

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
sel = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
sel.fit(Xtr, ytr)
cols = np.argsort(np.abs(sel.coef_[0]))[::-1][:nf]

probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr[:, cols], ytr)

def f1(X, y): return f1_score(y, probe.predict(X[:, cols]))
indist = f1(Xte, yte); rt = f1(Xrt, yrt); am = f1(Xam, yam)
mean_ood = (rt + am) / 2
print(f"indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={mean_ood:.4f}")
