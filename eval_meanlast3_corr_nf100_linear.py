import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling, nf = "meanlast3", 100

def X(split): return data[split][0][pooling].float().numpy()
def Y(split): return data[split][1].numpy()

Xtr, ytr = X("train"), Y("train")
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
Z = lambda A: (A - mu) / sd
Xtr_s = Z(Xtr)

# corr selection: top-nf by |point-biserial corr| with ytr
yc = ytr - ytr.mean()
Xc = Xtr_s - Xtr_s.mean(0)
num = (Xc * yc[:, None]).sum(0)
den = np.sqrt((Xc**2).sum(0) * (yc**2).sum()); den[den == 0] = 1.0
corr = num / den
cols = np.argsort(-np.abs(corr))[:nf]

# linear probe
probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_s[:, cols], ytr)

def f1(split):
    return f1_score(Y(split), probe.predict(Z(X(split))[:, cols]))

indist = f1("test"); rt = f1("rt"); am = f1("am")
mean_ood = (rt + am) / 2
print(indist, rt, am, mean_ood)
