import torch, numpy as np
from sklearn.feature_selection import f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "mean"; nf = 100

def get(split):
    P, y = data[split]
    return P[pooling].float().numpy(), np.asarray(y).astype(int)

Xtr, ytr = get("train")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

# corr selection: point-biserial = pearson corr between feature and binary label
yc = (ytr - ytr.mean())
corr = (Xtr_s * yc[:, None]).mean(0)  # std features, so this ~ point-biserial up to scale
# proper point-biserial on standardized X: corr = cov / (std_x*std_y); std_x=1
corr = corr / ytr.std()
cols = np.argsort(np.abs(corr))[::-1][:nf]

def feats(split):
    X, y = get(split)
    Xs = (X - mu) / sd
    return Xs[:, cols], y

Xtr_sel = Xtr_s[:, cols]
clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
clf.fit(Xtr_sel, ytr)

res = {}
for split, name in [("test","indist"),("rt","rt"),("am","am")]:
    Xs, y = feats(split)
    res[name] = f1_score(y, clf.predict(Xs))

res["mean_ood"] = (res["rt"] + res["am"]) / 2
print(res)
