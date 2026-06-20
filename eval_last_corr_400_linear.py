import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "last"; nf = 400

def getX(split): return data[split][0][pooling].float().numpy()
def gety(split): return data[split][1].numpy()

Xtr = getX("train"); ytr = gety("train")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

# corr selection: point-biserial = pearson corr with binary label
yc = ytr - ytr.mean()
num = (Xtr_s * yc[:, None]).sum(0)
den = np.sqrt((Xtr_s**2).sum(0) * (yc**2).sum())
den[den == 0] = 1.0
corr = np.abs(num / den)
cols = np.argsort(corr)[::-1][:nf]

clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr_s[:, cols], ytr)

def evalf1(split):
    X = (getX(split) - mu) / sd
    return f1_score(gety(split), clf.predict(X[:, cols]))

indist = evalf1("test"); rt = evalf1("rt"); am = evalf1("am")
mean_ood = (rt + am) / 2
print(f"indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={mean_ood:.4f}")
