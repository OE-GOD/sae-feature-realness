import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "max"; nf = 400

def getX(split): return data[split][0][pooling].float().numpy()
def gety(split): return data[split][1].numpy()

Xtr = getX("train"); ytr = gety("train")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

# corr selection: point-biserial = pearson corr with binary label
yc = ytr - ytr.mean()
xc = Xtr_s - Xtr_s.mean(0)
num = (xc * yc[:, None]).sum(0)
den = np.sqrt((xc**2).sum(0) * (yc**2).sum()) + 1e-12
corr = np.abs(num / den)
cols = np.argsort(-corr)[:nf]

probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_s[:, cols], ytr)

def ev(split):
    X = (getX(split) - mu) / sd
    return f1_score(gety(split), probe.predict(X[:, cols]))

indist = ev("test"); rt = ev("rt"); am = ev("am")
mean_ood = (rt + am) / 2
print(indist, rt, am, mean_ood)
