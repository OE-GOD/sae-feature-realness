import torch, numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "mean"; nf = 100

def get(split):
    P, y = data[split]
    return P[pooling].float().numpy(), y.numpy()

Xtr, ytr = get("train")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:nf]

Xtr_sel = Xtr_s[:, cols]
probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr_sel, ytr)

def evalf(split):
    X, y = get(split)
    Xs = (X - mu) / sd
    return f1_score(y, probe.predict(Xs[:, cols]))

indist = evalf("test"); rt = evalf("rt"); am = evalf("am")
mean_ood = (rt + am) / 2
print(f"INDIST={indist}\nRT={rt}\nAM={am}\nMEANOOD={mean_ood}")
