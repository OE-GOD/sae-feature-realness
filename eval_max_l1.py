import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling, nf = "max", 400

Xtr = data["train"][0][pooling].float().numpy()
ytr = data["train"][1].numpy()

mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

# l1 selection
sel = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
sel.fit(Xtr_s, ytr)
coef = np.abs(sel.coef_[0])
cols = np.argsort(coef)[::-1][:nf]

# probe: linear
probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_s[:, cols], ytr)

def evf(split):
    X = data[split][0][pooling].float().numpy()
    y = data[split][1].numpy()
    Xs = (X - mu) / sd
    pred = probe.predict(Xs[:, cols])
    return f1_score(y, pred)

indist = evf("test")
rt = evf("rt")
am = evf("am")
mean_ood = (rt + am) / 2
print(f"indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={mean_ood:.4f}")
