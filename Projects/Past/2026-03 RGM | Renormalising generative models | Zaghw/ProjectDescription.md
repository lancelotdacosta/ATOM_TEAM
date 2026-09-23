# Project proposal: Understanding Renormalizing Generative Models (RGMs)

## Background and motivation
Renormalizing Generative Models (RGMs) are a recently developed class of hierarchical, dyadic, discrete-state Bayesian generative models for time series. They are motivated by ideas from the renormalization group and aim to grow and learn *abstractions* online from time-series data based on successive *spatio-temporal coarse-grainings*. Preliminary empirical prior work suggests they can model across modalities (e.g., video and audio) and can support planning in simplified environments (e.g., Atari-style tasks).

Despite this promise, RGMs are currently poorly understood from a mathematical and statistical perspective. In particular, the original papers do not make explicit the underlying non-parametric Bayesian machinery. Preliminary explorations also suggest limitations in generalization. Finally, the primary implementation is currently in MATLAB (SPM).

This project aims to make RGMs more accessible (software), more transparent (theory), and better characterized (empirics).

## Goals (three threads)
1. **Open-source implementation (JAX):** produce a clean, well-documented, open-source implementation of core RGM routines in JAX, accompanied by educational notebooks.
2. **Mathematical understanding (tutorial write-up):** develop a precise, “tutorial paper”-style exposition of RGMs that makes the statistical and mathematical structure explicit and situates RGMs within non-parametric Bayesian modeling.
3. **Strengths and limitations (empirical investigation, time permitting):** design and implement targeted experiments to probe when/why RGMs work, when they break, and situate them within the existing ML landscape (e.g., imitation learning, hierarchical modeling and planning).

## Expected deliverables
A good “camera-ready” set of outputs would be:
- A public JAX codebase with: core model/data structures, inference/learning routines, tests, and 2–4 tutorial notebooks.
- A comprehensive technical document explaining RGMs with clear notation, model assumptions, and links to non-parametric Bayes.
- A focused empirical report summarizing experiments with RGM showing when/why they work and when they break, ideally also benchmarking with popular algorithms on 1-2 environments of interest, with discussion of broad positioning in the ML literature.

Depending on the results, this would be submitted as a workshop or conference paper.

## Suggested milestones (adjustable)
- **Weeks 1–4:** reproduce code from MATLAB/SPM implementation in JAX with tests checking behavior up to a significant amount of decimals with respect to MATLAB implementation on core environments. Understand code and write notebooks.
- **Weeks 5–8:** understand RGMs mathematically. Write a technical mathematical document relating them to non-parametric Bayes. Streamline the code.
- **Weeks 9–11:** run 1-2 experiments on popular environments. Assess what works and what breaks. Benchmark with popular baselines; write up results.

## Prerequisites / useful background
- Strong Python; comfort with JAX.
- Good knowledge of probability and Bayesian modeling; familiarity with variational inference/message passing, and ideally non-parametric Bayes.
- Interest in time series, hierarchical models, and planning / POMDPs / reinforcement learning.

## Supervision structure
- **Weekly:** 1-hour meeting, split into the following:
- **Every two weeks:** 1-hour meeting with Sanjeev Namjoshi (who has significant hands-on expertise with RGMs and prior experience translating them to JAX). More contact with Sanjeev would be encouraged time permitting on his side.
- **Every two weeks:** 1-hour meeting with colleagues working on hierarchical POMDPs, to cross-pollinate ideas relevant to hierarchical Bayesian learning and decision making.

Note: Sanjeev’s prior translation work is proprietary and cannot be transferred directly, but his guidance should substantially accelerate progress.

## References on RGM
1. Friston, K., Heins, C., Verbelen, T., Da Costa, L., Salvatori, T., et al. (2024). *From pixels to planning: scale-free active inference*. arXiv:2407.20292.
2. Friston, K., Parr, T., Heins, C., Da Costa, L., Salvatori, T., et al. (2025). *Gradient-Free De Novo Learning*. Entropy, 27(9).
3. Friston, K., Da Costa, L., Tschantz, A., Heins, C., Buckley, C., Verbelen, T., Parr, T. (2025). *Active inference and artificial reasoning*. arXiv:2512.21129.

Note: RGMs have been developed chronologically in this set of papers. The goal would be to work with the latest version from [3] available in the SPM Academic software, the latest development version I can share.
