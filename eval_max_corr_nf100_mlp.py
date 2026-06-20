import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)

pooling = "max"
nf = 100

def getX(split):
    return data[split][0][pooling].float().numpy()

Xtr = getX("train"); ytr = data["train"][1].numpy()
Xte = getX("test"); yte = data["test"][1].numpy()
Xrt = getX("rt"); yrt = data["rt"][1].numpy()
Xam = getX("am"); yam = data["am"][1].numpy()

mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Ztr = (Xtr - mu) / sd
Zte = (Xte - mu) / sd
Zrt = (Xrt - mu) / sd
Zam = (Xam - mu) / sd

# corr selection: point-biserial = pearson corr with binary label
yc = ytr - ytr.mean()
denom = (Ztr.std(0) * yc.std())
denom[denom == 0] = 1.0
corr = (Ztr * yc[:, None]).mean(0) / denom
cols = np.argsort(-np.abs(corr))[:nf]

Str = Ztr[:, cols]; Ste = Zte[:, cols]; Srt = Zrt[:, cols]; Sam = Zam[:, cols]

clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
clf.fit(Str, ytr)

indist_f1 = f1_score(yte, clf.predict(Ste))
rt_f1 = f1_score(yrt, clf.predict(Srt))
am_f1 = f1_score(yam, clf.predict(Sam))
mean_ood = (rt_f1 + am_f1) / 2

print(f"indist_f1={indist_f1:.4f}")
print(f"rt_f1={rt_f1:.4f}")
print(f"am_f1={am_f1:.4f}")
print(f"mean_ood_f1={mean_ood:.4f}")
