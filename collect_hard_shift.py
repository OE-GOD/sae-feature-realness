"""Collect HARDER-shift sentiment domains to test the frontier caveat: does two-view
disagreement still win when the shift is semantic/register, not just review-vocabulary?
Domains far from movie/product reviews: tweets (informal), financial news (formal),
poems (literary). Same pipeline as collect_more_ood.py: Gemma Scope L12, mean+max pool."""
import os, torch
os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
dev = "mps" if torch.backends.mps.is_available() else "cpu"
sae = SAE.from_pretrained("gemma-scope-2b-pt-res", "layer_12/width_16k/average_l0_82", device=dev)
hook = sae.cfg.metadata["hook_name"]
model = HookedTransformer.from_pretrained("gemma-2-2b", device=dev, dtype=torch.bfloat16)

def pools(text):
    ids = model.to_tokens(text[:300])
    with torch.no_grad():
        _, c = model.run_with_cache(ids, names_filter=[hook]); f = sae.encode(c[hook][0]).float()
    return f.mean(0).half().cpu(), f.max(0).values.half().cpu()

def collect(texts):
    M = []; X = []
    for t in texts:
        if not t or len(t) < 5:
            M.append(torch.zeros(16384, dtype=torch.half)); X.append(torch.zeros(16384, dtype=torch.half)); continue
        m, x = pools(t); M.append(m); X.append(x)
        if dev == "mps": torch.mps.empty_cache()
    return {"mean": torch.stack(M), "max": torch.stack(X)}

def balanced(texts, labels, n):
    """labels already mapped to {0,1}; take first n of each class."""
    pos = [t for t, l in zip(texts, labels) if l == 1][:n]
    neg = [t for t, l in zip(texts, labels) if l == 0][:n]
    return neg + pos, torch.tensor([0] * len(neg) + [1] * len(pos))

N = 120
out = {}

# tweets: tweet_eval sentiment (0 neg, 1 neutral, 2 pos) -> keep {0,2}
try:
    ds = load_dataset("tweet_eval", "sentiment", split="test")
    txt = [r["text"] for r in ds if r["label"] in (0, 2)]
    lab = [1 if r["label"] == 2 else 0 for r in ds if r["label"] in (0, 2)]
    t, y = balanced(txt, lab, N); out["tweet"] = (collect(t), y); print("tweet done", len(y))
except Exception as e:
    print("tweet fail", repr(e))

# financial news: financial_phrasebank (0 neg, 1 neutral, 2 pos) -> keep {0,2}
try:
    ds = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
    txt = [r["sentence"] for r in ds if r["label"] in (0, 2)]
    lab = [1 if r["label"] == 2 else 0 for r in ds if r["label"] in (0, 2)]
    t, y = balanced(txt, lab, N); out["financial"] = (collect(t), y); print("financial done", len(y))
except Exception as e:
    print("financial fail", repr(e))

# poems: poem_sentiment (0 neg, 1 pos, 2 no_impact, 3 mixed) -> keep {0,1}
try:
    ds = load_dataset("poem_sentiment", split="train")
    txt = [r["verse_text"] for r in ds if r["label"] in (0, 1)]
    lab = [r["label"] for r in ds if r["label"] in (0, 1)]
    t, y = balanced(txt, lab, N); out["poem"] = (collect(t), y); print("poem done", len(y))
except Exception as e:
    print("poem fail", repr(e))

torch.save(out, "/Users/oe/rebuild/hard_shift_sentiment.pt")
for k, (P, y) in out.items():
    print(f"{k}: {P['mean'].shape[0]} examples, pos={int(y.sum())}")
print("saved hard_shift_sentiment.pt")
