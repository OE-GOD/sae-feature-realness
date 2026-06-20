# The Rebuild — A Recap (read this anytime it gets fuzzy)

*Written after rebuilding my SAE research from scratch, by hand, June 2026.
Everything here I either built, derived, or taught back myself.*

---

## Why I did this

I had a research portfolio built with AI doing most of the typing. Then one
interview asked me one question and I blanked — because I could *recognize* my
work but not *produce* it. Recognition ≠ understanding. So I rebuilt the core
of my research from zero, typing every line, explaining every step before it ran.
This is the map of what I found.

---

## 1. The founding problem: the model's thought is a smoothie

A language model reads text. Inside, every token becomes a list of **768 numbers**
— the *activation*. By layer 6 that list holds everything the model understands so
far: topic, tense, tone, all of it.

But you **can't read it**. Not because there are many numbers — because every
concept is **smeared across all 768 slots at once.** No single slot means
"interview." It's a smoothie: banana + spinach + milk all blended, no banana
visible anywhere.

- **Weights** = the frozen training, loaded onto disk. Same for everyone. The instrument.
- **Activations** = born only while reading a sentence. Change with the input. The thought.
- The thought lives in the **activations**, which is why you must RUN the model
  (and `run_with_cache`, because a normal run throws the middle away).
- The **residual stream** = a running sum: every layer ADDS its note, nothing
  erases. It's the model's only wire. Read it = read everything so far.

**The SAE exists to un-blend the smoothie into a readable recipe.**

---

## 2. What an SAE is: scorer + bouncer + rebuilder

Three parts:

- **Encoder (the scorer):** reads the 768-number thought, scores all ~2,048
  possible ingredients. "How much dog-ness? past-tense-ness? interview-ness?"
- **TopK (the bouncer):** keeps only the top k scores (mine: 32), zeros the rest.
  *Why few?* Because a 2,048-number description is just a second smoothie.
  Few survivors = a short, readable list. Readability is the whole product.
- **Decoder (the rebuilder):** each feature owns a "recipe card" — a 768-number
  direction. Rebuilds the thought = sum of (strength × card) over survivors.

**A feature = one column of the decoder = a direction in activation space.**
It *fires* when its concept appears in the input. Training pushes the
**reconstruction error down** (mine: 0.63 → 0.11 over 5 epochs).

### The k dial (a core tension of the field)
- k UP → better reconstruction, worse readability (more survivors = more goo)
- k DOWN → cleaner recipe, lossier rebuild
It trades against itself. That's why "variance explained 0.98!" never proves an
SAE is good on its own.

### Sparsity economics (explains everything downstream)
Few slots → slots are scarce → a recurring pattern earns its own feature because
one slot buys near-perfect reconstruction of every occurrence. This single
principle explains feature splitting, absorption, composition — and the
hyper-specific junk features (my feature 989 fired only on the period inside
`<name>PD1.x</name>` XML tags, because one weird file repeated it dozens of times).

---

## 3. The experiment that matters: most features aren't real

I trained **two** SAEs. Held the SAME: food (same activations), width,
architecture. Changed ONE thing: the random **seed** (the starting numbers).

Result:
- Both reached **identical** loss: 0.1056 = 0.1056. Equally good at the job.
- Yet only **1.0%** of features had a twin (cosine > 0.9) in the other SAE.
- Median best-match cosine: **0.22**. Most features had no counterpart at all.

### The logic (this is the part that licenses the conclusion)
> A difference can only be caused by something that was different.
> The thoughts didn't differ; only the seed did.
> So the disagreeing features come from the **seed**, not the **model** —
> and that's what makes them **training artifacts.**

**Artifact** = something the *instrument* created, mistaken for something
*discovered*. Like lens flare in a photo: it's in the camera, not the sky.

### Why a seed produces different features (the hikers)
Two hikers dropped at different random spots both walk downhill to sea level
(both rebuild well) — but land in **different valleys** (different dictionaries).
This only happens because there are **many equally-good dictionaries**. If the
data forced ONE answer, every seed would find it and all SAEs would agree.
The disagreement is proof the data allows many stories; the seed picks which one.

### The test, stated generally (a realness detector)
> Vary the instrument's accidents. Hold the object fixed. See what survives.
> Survives every seed → in the model → real.  Vanishes → in the seed → artifact.

This is the whole reason metrics mislead: variance explained, CE-delta,
monosemanticity all measure *whether the SAE does its job* — none measure
*whether the features are real.* (My old 16k SAE: great metrics, only 2.14%
replicated across four conditions.)

---

## 4. The twist: "real" has TWO grades, and they're independent axes

A feature being *found by every seed* (stable) is different from the model
actually *using* it. Test "used" by **ablation**: cut the feature's contribution
out of the thought (subtract strength × its decoder column), let it flow to the
output, see if the prediction moves.

But a raw move means nothing without a **control**: cut a *different* feature,
same surgery, and measure that too. The control defines the **noise band**.

My ablation: IOI prompt, base P(" Mary") = 0.57. Cut the LOUDEST-firing feature
(1853, activation 9.78) → moved −0.03. Control feature → moved +0.04.
The target sat **inside** the noise band → it fires loud but doesn't drive.

**Loud ≠ important.** A speedometer needle swings hard but ripping it out doesn't
slow the car — it *reads* the speed, doesn't *cause* it. Firing ≠ driving.
The only way to tell them apart is to cut it and watch.

### The 2×2 (the research-grade insight)
Stability and causation are **orthogonal axes**, not a ladder:

```
                STABLE              NOT STABLE
USED        real driver         manifold feature   ← used but seed-dependent
            (atom that drives)   (my newlines: cosine 0.4, but ablation = +15 nats!)
NOT USED    thermometer         pure artifact
            (stable but idle)   (seed noise, idle)
```

The top-right box is the subtle one: a feature can be **causally real but
individually fake** — the model uses the *region* (a river the city depends on),
but the SAE's *slicing* of it into features is one arbitrary map among many
(where you draw the district lines is a seed-choice). My manifold/newline
features live here. "Stability and importance are orthogonal axes."

---

## 5. The 9 core hooks (everything else is lookup)

1. Weights = frozen instrument. Activations = the thought (born from input).
2. Residual stream = running sum; the model's only wire.
3. Activations are unreadable mush — concepts smeared across all slots. (Why SAEs exist.)
4. SAE = encoder (scorer) + TopK (bouncer) + decoder (the features = directions).
5. Sparsity economics: recurring patterns earn slots. (Explains the disease family.)
6. Detect ≠ cause. Firing ≠ driving. Test by ablation.
7. The causal test: change something inside, watch the output. (Patching → AtP → AtP*.)
8. Trust nothing a method says about itself. Independent test or it didn't happen.
9. The method: predict → run → compare → update.

---

## 6. What I actually learned

About SAEs: most features are training artifacts; metrics measure the job, not
realness; stability and causation are different axes.

About myself: I can learn this. Three days after blanking in an interview and
saying "I belong in low wage," I built the whole stack by hand, taught all three
lessons back to a skeptic, and asked a question sharp enough to rebuild the
framework into a 2×2 that matched my own published finding. That's not
recognition. That's understanding. The difference is everything.

---

*Files: 01_look → 06_ablate in /Users/oe/rebuild/. Figures: fig1–fig3.
The numbers in this doc are from my own runs. Re-read when it gets fuzzy.*
