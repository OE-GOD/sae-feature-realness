import torch, numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling, nf = "mean", 400

Xtr = data["train"][0][pooling].float().numpy()
ytr = data["train"][1].numpy()

mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

mi = mutual_info_classif(Xtr_s, ytr, random_state=0)
cols = np.argsort(mi)[::-1][:nf]

probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_s[:, cols], ytr)

def ev(split):
    X = data[split][0][pooling].float().numpy()
    y = data[split][1].numpy()
    Xs = ((X - mu) / sd)[:, cols]
    return f1_score(y, probe.predict(Xs))

indist = ev("test"); rt = ev("rt"); am = ev("am")
mean_ood = (rt + am) / 2
print(f"RESULT {indist:.6f} {rt:.6f} {am:.6f} {mean_ood:.6f}")
