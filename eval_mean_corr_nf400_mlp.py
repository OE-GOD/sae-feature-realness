import torch, numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "mean"; nf = 400

def get(split):
    P, y = data[split]
    return P[pooling].float().numpy(), y.numpy()

Xtr, ytr = get("train")
mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

# corr selection: point-biserial = pearson corr with binary label
yc = (ytr - ytr.mean())
corr = (Xtr_s * yc[:, None]).mean(0)  # since Xtr_s standardized, this is proportional to corr
# proper point-biserial on standardized features:
corr_vals = np.array([np.corrcoef(Xtr_s[:, j], ytr)[0, 1] for j in range(Xtr_s.shape[1])])
corr_vals = np.nan_to_num(corr_vals)
cols = np.argsort(-np.abs(corr_vals))[:nf]

Xtr_sel = Xtr_s[:, cols]
probe = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
probe.fit(Xtr_sel, ytr)

def evalf1(split):
    X, y = get(split)
    Xs = (X - mu) / sd
    return f1_score(y, probe.predict(Xs[:, cols]))

indist = evalf1("test")
rt = evalf1("rt")
am = evalf1("am")
mean_ood = (rt + am) / 2
print(f"indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={mean_ood:.4f}")
