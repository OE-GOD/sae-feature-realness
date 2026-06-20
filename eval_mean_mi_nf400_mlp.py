import torch, numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "mean"; nf = 400

def getX(split):
    return data[split][0][pooling].float().numpy()
def gety(split):
    return data[split][1].numpy()

Xtr = getX("train"); ytr = gety("train")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:nf]

Xtr_sel = Xtr_s[:, cols]
probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr_sel, ytr)

def evalsplit(split):
    X = (getX(split) - mu) / sd
    return f1_score(gety(split), probe.predict(X[:, cols]))

indist = evalsplit("test")
rt = evalsplit("rt")
am = evalsplit("am")
mean_ood = (rt + am) / 2
print(f"indist_f1={indist:.4f}")
print(f"rt_f1={rt:.4f}")
print(f"am_f1={am:.4f}")
print(f"mean_ood_f1={mean_ood:.4f}")
