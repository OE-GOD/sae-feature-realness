import torch, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling = "last"; nf = 100

Xtr = data["train"][0][pooling].float().numpy()
ytr = data["train"][1].numpy()

mean = Xtr.mean(0); std = Xtr.std(0); std[std == 0] = 1.0
Xtr_s = (Xtr - mean) / std

# corr selection: point-biserial == pearson corr with binary y
yc = ytr - ytr.mean()
num = (Xtr_s * yc[:, None]).sum(0)
den = np.sqrt((Xtr_s**2).sum(0) * (yc**2).sum())
den[den == 0] = 1.0
corr = num / den
cols = np.argsort(np.abs(corr))[::-1][:nf]

Xtr_sel = Xtr_s[:, cols]

clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=400, random_state=0)
clf.fit(Xtr_sel, ytr)

def ev(split):
    X = data[split][0][pooling].float().numpy()
    y = data[split][1].numpy()
    Xs = ((X - mean) / std)[:, cols]
    return f1_score(y, clf.predict(Xs))

indist = ev("test"); rt = ev("rt"); am = ev("am")
mean_ood = (rt + am) / 2
print(f"{indist:.4f} {rt:.4f} {am:.4f} {mean_ood:.4f}")
