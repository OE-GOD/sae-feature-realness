---
title: "Can You Tell a Real SAE Feature From a Fake One — Cheaply?"
subtitle: "I rebuilt my interpretability research from scratch, then ran a small experiment on a problem I picked myself. Here's an honest result."
date: 2026-06-15
---

## Why I did this

A few weeks ago I had a research portfolio I couldn't fully defend — AI had done
most of the typing, and I could *recognize* my own work without being able to
*reproduce* it. So I rebuilt the core of it by hand: trained a sparse autoencoder
(SAE) on a language model from scratch, typing every line, explaining every step
before running it. Then I did the thing that actually makes you a researcher
instead of a student of one — I picked my own problem and ran an experiment
nobody had run on my setup. This post is that experiment, reported honestly,
including what's weak about it.

## The problem: are SAE features even real?

SAEs are the standard tool for reading a language model's "thoughts." A model's
internal activation is an unreadable list of 768 numbers with every concept smeared
across all of them; an SAE un-blends that into a short list of named features. The
whole field then builds on those features — circuits, interpretations, safety
arguments like "watch feature X to detect deception."

But there's a load-bearing assumption underneath all of it: **that the features
are real** — genuine parts of how the model computes, not artifacts the SAE's own
training invented.

I'd already found (on a 16k-feature SAE) that this assumption is shaky: train two
SAEs on the same model, changing only the random seed, and only ~2% of features
replicate. Most "features" are training artifacts — they exist in one seed and
vanish in the next. (Rebuilt at toy scale here, I got ~1% at cosine > 0.9, with
both SAEs reaching *identical* reconstruction loss — equally good, almost no shared
vocabulary.)

The catch: checking replication is **expensive**. You have to train a second SAE.
So I asked a smaller, more useful question:

> **Can I predict whether a feature is real from a single SAE — cheaply?**

## The hypothesis

The cheapest signal available from one SAE is **how often each feature fires.**
My prior work suggested frequent features tend to be the stable, real ones and
rare features the unstable junk. So I predicted: **firing frequency strongly
predicts stability.**

## Method (deliberately small)

- SAE: 2,048 features, TopK k=32, trained on layer-6 activations of Pythia-160M
  (~70k tokens from The Pile).
- A second SAE, identical except for the random seed, as the stability ground truth.
- **Stability** of each feature = best cosine similarity of its decoder direction
  to any feature in the second SAE.
- **Frequency** of each feature = fraction of tokens it fires on (cheap, one SAE).
- Correlate the two.

## Result (and where my prediction was wrong)

```
Pearson(raw frequency, stability)     = 0.06   ← basically nothing
Pearson(log-frequency, stability)     = 0.52   ← moderate

stability by frequency quartile:
  Q1 rarest    : 0.22
  Q2           : 0.48
  Q3 sweet spot: 0.54   ← peak
  Q4 commonest : 0.48   ← dips back down
```

Two things I didn't predict:

**1. The signal only appears on a log scale.** Firing frequencies span orders of
magnitude (a rare feature fires on ~0.0002 of tokens, a common one on ~0.3). On a
raw scale every rare feature looks like zero, and the correlation is blind (0.06).
Log-spacing reveals the relationship (0.52). *How you look decides what you see* —
the raw view would have told me "no relationship," and I'd have been wrong.

**2. It's an inverted-U, not a straight line.** Stability rises sharply with
frequency, peaks in the upper-middle, then *dips* at the most common features. So
my prediction was only **partly** right: frequency carries a real, cheap signal —
but it's moderate, and "more frequent = more stable" is false at the top end.

## Why the inverted-U? (staring at the actual features)

A correlation isn't an explanation, so I read the transcripts — the top-firing
tokens for one feature in each region:

- **Q1 rarest, unstable (stability 0.20):** fires on the punctuation inside
  statistical p-value notation — "(P < 0.05)", "(P < 0.01)". A hyper-specific
  fragment. Too narrow; each seed carves the tiny niche differently.

- **Q3 peak, stable (stability 0.97):** fires cleanly on the word **"as"** —
  "as well as", "as a result", "as soon as". A crisp, consistent concept. Every
  seed finds it the same way.

- **Q4 commonest, dips (fires on ~90% of tokens, stability 0.45):** fires on a
  grab-bag — get, Cent, Meet, 's, argues, natural, IS. **No concept at all.** A
  feature that's almost always on isn't detecting anything; it's a near-constant
  background direction, and each seed splits that "everything" differently.

**The finding:** stable features are clean concepts, and clean concepts live in
the *middle* of the frequency range. Both extremes replicate worse, for opposite
reasons — rare features are too narrow (hyper-specific fragments), the commonest
are too broad (incoherent always-on directions).

## Limitations (the honest part)

This is a toy result and I won't pretend otherwise:

- One seed-pair, 2,048 features, a 160M model, ~70k tokens. Small everything.
- Only ~2 features in the sample were actually "stable" (cos > 0.9) — the stable
  region is barely sampled, because stable features are rare (that's the whole
  point, and it bites here as a sampling problem).
- All causal effects in my related ablation runs were tiny (~0.006 loss), so
  signal-to-noise is low.
- A real version needs multiple seeds, a production SAE (e.g. Gemma Scope), and
  ideally a downstream task to ground "real" in something that matters.

So the honest claim is narrow: *in a toy SAE, cheap firing-frequency partially
predicts expensive feature-stability, via an inverted-U, and the shape has a
transcript-level explanation.* Not "I solved cheap realness detection." A swing
at an open problem, at small scale.

## Why it matters

You can't understand how a model thinks using features that aren't real — so
telling real features from artifacts is the ground floor of interpretability. Right
now that test is expensive (train more SAEs). Any cheap proxy — even a partial one
like frequency — is a step toward auditing features before you trust them in a
circuit or a safety claim. Frequency alone isn't enough, but the inverted-U says
something usable: **be most suspicious of features at the frequency extremes.**

## What I actually learned

The result is small. The bigger thing: I picked a problem myself (instead of
absorbing one), predicted before running, read my own result without overclaiming
("partly," not "fully"), and explained a surprising curve by reading transcripts
instead of theorizing. That loop — not any single number — is the job.

*Code and figures: [link]. Built by hand; every line defensible.*
