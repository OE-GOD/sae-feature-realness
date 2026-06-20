import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset

model = HookedTransformer.from_pretrained("pythia-160m")
data = load_dataset("NeelNanda/pile-10k", split="train")
print("CHUNK 1: OK — model + 10000 docs")

vectors = []
for i in range(200):
    text = data[i]["text"][:2000]
    with torch.no_grad():
        logits, cache = model.run_with_cache(text)
    thought = cache["blocks.6.hook_resid_post"]
    vectors.append(thought[0].cpu())
    if (i + 1) % 50 == 0:
        print(f"  ...{i+1}/200 documents eaten")

pile = torch.cat(vectors)
torch.save(pile, "/Users/oe/rebuild/thoughts.pt")
print("PILE SHAPE:", tuple(pile.shape))
print("saved to /Users/oe/rebuild/thoughts.pt")
