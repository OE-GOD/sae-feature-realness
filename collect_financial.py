"""Replace the failed financial_phrasebank (script-based, now blocked) with a parquet-native
financial-sentiment dataset: zeroshot/twitter-financial-news-sentiment (0=Bearish, 1=Bullish,
2=Neutral). Financial headlines = formal, specialized register, far from reviews. Merge into
hard_shift_sentiment.pt."""
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

N = 120
out = torch.load("/Users/oe/rebuild/hard_shift_sentiment.pt", weights_only=False)
try:
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    txt = [r["text"] for r in ds if r["label"] in (0, 1)]
    lab = [r["label"] for r in ds if r["label"] in (0, 1)]   # 0 bearish->neg, 1 bullish->pos
    pos = [t for t, l in zip(txt, lab) if l == 1][:N]; neg = [t for t, l in zip(txt, lab) if l == 0][:N]
    t = neg + pos; y = torch.tensor([0] * len(neg) + [1] * len(pos))
    out["financial"] = (collect(t), y); print("financial done", len(y))
except Exception as e:
    print("financial fail", repr(e))
torch.save(out, "/Users/oe/rebuild/hard_shift_sentiment.pt")
for k, (P, yy) in out.items():
    print(f"{k}: {P['mean'].shape[0]} examples, pos={int(yy.sum())}")
print("saved")
