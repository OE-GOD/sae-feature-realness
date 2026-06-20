import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = torch.load("/Users/oe/rebuild/sem_pooling_dataset.pt", weights_only=False)
pooling, nf = "last", 100

def get(split):
    P, y = data[split]
    return P[pooling].float().numpy(), y.numpy()

Xtr, ytr = get("train")
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
Xtr_s = (Xtr - mu) / sd

# corr selection: point-biserial = pearson corr with binary label
yc = ytr - ytr.mean()
corr = (Xtr_s * yc[:, None]).mean(0) / (Xtr_s.std(0) * yc.std() + 1e-12)
cols = np.argsort(np.abs(corr))[::-1][:nf]

probe = LogisticRegression(max_iter=1000)
probe.fit(Xtr_s[:, cols], ytr)

def f1_on(split):
    X, y = get(split)
    Xs = ((X - mu) / sd)[:, cols]
    return f1_score(y, probe.predict(Xs))

indist = f1_on("test"); rt = f1_on("rt"); am = f1_on("am")
mean_ood = (rt + am) / 2
print(f"indist={indist:.4f} rt={rt:.4f} am={am:.4f} mean_ood={mean_ood:.4f}")
