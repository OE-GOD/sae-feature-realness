from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("pythia-160m")
text = "The interview did not go well today"
logits, cache = model.run_with_cache(text)
thought = cache["blocks.6.hook_resid_post"]
print("SHAPE:", tuple(thought.shape))
print()
print("the model's thought at layer 6, token 'today', first 20 of 768 numbers:")
print(thought[0, -1, :20])
