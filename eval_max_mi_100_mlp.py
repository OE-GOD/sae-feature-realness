import torch, numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "max"

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

nf = 100
mi = mutual_info_classif(Xtr, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:nf]

clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
clf.fit(Xtr[:, cols], ytr)

f1_te = f1_score(yte, clf.predict(Xte[:, cols]))
f1_rt = f1_score(yrt, clf.predict(Xrt[:, cols]))
f1_am = f1_score(yam, clf.predict(Xam[:, cols]))
mean_ood = (f1_rt + f1_am) / 2
print(f"indist {f1_te:.4f} rt {f1_rt:.4f} am {f1_am:.4f} mean_ood {mean_ood:.4f}")
