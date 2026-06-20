import torch
import torch.nn as nn
import json

torch.manual_seed(0)
d = torch.load('/Users/oe/rebuild/detector_dataset.pt')
keys = d['keys']
train_F = d['train_F'].float()
test_F = d['test_F'].float()
ood_F = d['ood_F'].float()

N_FEATURES = 10

def select_meandiff(F, y, k):
    pos = F[y == 1]
    neg = F[y == 0]
    score = (pos.mean(0) - neg.mean(0)).abs()
    return torch.topk(score, k).indices

def best_f1_threshold(scores, y):
    order = torch.argsort(scores, descending=True)
    ys = y[order]
    P = y.sum().item()
    if P == 0:
        return 0.5
    tp = torch.cumsum(ys, 0)
    fp = torch.cumsum(1 - ys, 0)
    precision = tp / (tp + fp)
    recall = tp / P
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    bi = torch.argmax(f1).item()
    return scores[order][bi].item()

def f1_at(scores, y, thr):
    pred = (scores >= thr).float()
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)

class MLP(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

results = []
for C in keys:
    ytr = d['train_L'][C].float()
    if ytr.sum() < 5:
        continue
    yte = d['test_L'][C].float()
    yood = d['ood_L'][C].float()

    cols = select_meandiff(train_F, ytr, N_FEATURES)
    Xtr = train_F[:, cols]
    Xte = test_F[:, cols]
    Xood = ood_F[:, cols]

    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd
    Xood_s = (Xood - mu) / sd

    torch.manual_seed(0)
    model = MLP(N_FEATURES)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    lossfn = nn.BCEWithLogitsLoss()
    for ep in range(300):
        opt.zero_grad()
        loss = lossfn(model(Xtr_s), ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        ptr = torch.sigmoid(model(Xtr_s))
        pte = torch.sigmoid(model(Xte_s))
        pood = torch.sigmoid(model(Xood_s))

    thr = best_f1_threshold(ptr, ytr)
    indist = f1_at(pte, yte, thr)
    ood = f1_at(pood, yood, thr)
    results.append((C, indist, ood))
    print(f"{C}: indist={indist:.4f} ood={ood:.4f}")

mean_ind = sum(r[1] for r in results) / len(results)
mean_ood = sum(r[2] for r in results) / len(results)
print(f"MEAN indist={mean_ind:.4f} ood={mean_ood:.4f}")
print("JSON", json.dumps({"mean_indist_f1": mean_ind, "mean_ood_f1": mean_ood,
    "per_concept": [{"concept": c, "indist_f1": i, "ood_f1": o} for c, i, o in results]}))
