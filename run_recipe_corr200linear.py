import torch, json
torch.manual_seed(0)

PATH = "/Users/oe/rebuild/gemma_detector_dataset.pt"
N_FEATURES = 200

d = torch.load(PATH, map_location="cpu")
concepts = ["newline","comma","period","digit","space_pre","cap_start"]

train_F = d["train_F"]
test_F  = d["test_F"]
ts_F    = d["ts_F"]
wk_F    = d["wk_F"]
train_L = d["train_L"]; test_L = d["test_L"]; ts_L = d["ts_L"]; wk_L = d["wk_L"]


def f1_at(scores, y, thr):
    pred = (scores >= thr).float()
    tp = (pred * y).sum(); fp = (pred * (1 - y)).sum(); fn = ((1 - pred) * y).sum()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp / denom).item()


def best_threshold(scores, y):
    qs = torch.linspace(0, 1, 1000)
    cands = torch.unique(torch.quantile(scores, qs))
    best_f1, best_thr = -1.0, 0.0
    for thr in cands:
        f = f1_at(scores, y, thr.item())
        if f > best_f1:
            best_f1, best_thr = f, thr.item()
    return best_thr


def train_logreg(X, y, wd=1e-4):
    dft = X.shape[1]
    w = torch.zeros(dft, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=300,
                            line_search_fn="strong_wolfe")
    pos = y.sum(); neg = (1 - y).sum()
    pw = (neg / pos.clamp(min=1)).clamp(max=50.0)
    weight = torch.where(y > 0.5, pw, torch.ones_like(y))

    def closure():
        opt.zero_grad()
        logits = X @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, weight=weight)
        loss = loss + wd * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


indist_list, ood_list, per_concept = [], [], []

for C in concepts:
    ytr = train_L[C].float()
    if ytr.sum() < 5:
        continue
    Xtr = train_F.float()
    xm = Xtr.mean(0); xs = Xtr.std(0).clamp(min=1e-6)
    ym = ytr.mean(); ysd = ytr.std().clamp(min=1e-6)
    cov = ((Xtr - xm) * (ytr - ym).unsqueeze(1)).mean(0)
    corr = torch.nan_to_num(cov / (xs * ysd), 0.0)
    sel = torch.topk(corr.abs(), N_FEATURES).indices
    mu = xm[sel]; sd = xs[sel]

    def prep(F):
        return (F[:, sel].float() - mu) / sd

    w, b = train_logreg(prep(train_F), ytr)

    def score(F):
        return prep(F) @ w + b

    thr = best_threshold(score(train_F), ytr)

    indist_f1 = f1_at(score(test_F), test_L[C].float(), thr)
    entry = {"concept": C, "indist_f1": indist_f1}
    indist_list.append(indist_f1)

    ts_y = ts_L[C].float()
    if ts_y.sum() >= 3:
        ts_f1 = f1_at(score(ts_F), ts_y, thr)
        entry["ts_f1"] = ts_f1; ood_list.append(ts_f1)
    wk_y = wk_L[C].float()
    if wk_y.sum() >= 3:
        wk_f1 = f1_at(score(wk_F), wk_y, thr)
        entry["wk_f1"] = wk_f1; ood_list.append(wk_f1)

    per_concept.append(entry)
    print(C, entry, flush=True)

out = {
    "recipe": "corr|nf200|linear",
    "mean_indist_f1": sum(indist_list) / len(indist_list),
    "mean_ood_f1": sum(ood_list) / len(ood_list),
    "per_concept": per_concept,
}
print(json.dumps(out, indent=2))
with open("/Users/oe/rebuild/recipe_result_corr200linear.json", "w") as f:
    json.dump(out, f, indent=2)
