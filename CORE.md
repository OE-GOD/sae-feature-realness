# THE CORE MAP — 9 things, everything else is lookup
(written 2026-06-12, after Steps 1-3 of the rebuild)

1. **Weights vs activations.** Weights = frozen training, loaded and waiting.
   Activations = born from the input, change with the input. The thought lives in activations.

2. **Residual stream = running sum.** Every layer ADDS its note, nothing erases.
   It is the model's only communication wire. Read it = read everything so far.

3. **Activations are unreadable mush.** Concepts smeared across 768 numbers.
   This is the founding problem. It is WHY SAEs exist.

4. **SAE = question + bouncer + substance.**
   Encoder row = the question ("how much of this ingredient?").
   TopK = the bouncer (few slots → short, readable list).
   Decoder column = the substance — the feature itself, a direction in activation space.

5. **Sparsity economics: recurring patterns earn slots.**
   Explains why features mean things. Also explains the diseases:
   splitting, absorption, composition — and my hyper-specific feature 989.

6. **Detect ≠ cause.** Firing = the model noticed. Only intervention shows the model USED it.
   My own finding: ~60% of labeled features are thermometers, not drivers.

7. **The causal test: change something inside, watch the output.**
   Activation patching = gold standard. AtP = gradient shortcut.
   Shortcuts break on flat curves (softmax saturation) and deep stacks (Gemma 0.41).

8. **Trust nothing a method says about itself.**
   Great metrics ≠ real features (my 2.14% replication result).
   Independent test — replication, coverage — or it didn't happen.

9. **The method: predict → run → compare → update.**
   Every experiment, every time. This is the whole job.

---
Bucket 2 (LOOKUP — forgetting is fine): API names, cache keys, shapes, hyperparams, paper details.
Bucket 3 (HANDS — built, not memorized): /Users/oe/rebuild/ Steps 1-3. Next: Step 4 (second seed,
cosine match, my own replication rate), Step 5 (ablate a feature, watch the output).
