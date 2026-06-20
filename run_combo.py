import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "meanlast3"; nf = 400

def get(split):
    P, y = data[split]
    return P[pooling].float().numpy(), y.numpy()

Xtr, ytr = get("train")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

# l1 selection
sel = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
sel.fit(Xtr_s, ytr)
cols = np.argsort(np.abs(sel.coef_[0]))[::-1][:nf]

probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_s[:, cols], ytr)

def evalf(split):
    X, y = get(split)
    Xs = (X - mu) / sd
    return f1_score(y, probe.predict(Xs[:, cols]))

indist = evalf("test"); rt = evalf("rt"); am = evalf("am")
mean_ood = (rt + am) / 2
print(indist, rt, am, mean_ood)
