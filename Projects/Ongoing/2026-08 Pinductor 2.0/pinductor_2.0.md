# Pinductor 2.0: beyond `Learning POMDP World Models from Observations with Language-Model Priors'

The goal of this project is to provide the next generation to the Pinductor agent. The target will be strong performance in benchmarks like ARC-AGI-3, which measure efficient reinforcement learning under strong sample and compute constraints, or world modeling benchmarks.

As a reminder, Pinductor is an RL agent that engages in active world modeling, that is, it builds a model of the world in a sample-efficient way by interacting with the world and using an LLM prior. It can then use this to solve RL tasks.

## Goal

**The goal of this project is strong performance in efficient RL under strong sample and compute constraints, or active world modeling benchmarks.**

These are the candidate benchmarks to test and evaluate the method on. The goal is to find improvements on Pinductor that will give us better performance across the board.
The goal of this project is strong performance on a benchmark while remaining relatively principled in the sense that we want to keep the architecture general so that it doesn't obviously overfit to the benchmarks that we choose.

### Benchmarks
Efficient RL benchmarks on the strong sample and compute constraints:
- [ARC-AGI-3](https://arcprize.org/arc-agi/3)
- The Atom benchmark, currently under development, will introduce a suite of environments and measure learning gains per sample and per unit of compute.
- Any RL environment where you measure performance per sample and per unit of compute (in an x,y,z axis). For new environments, one could consider
  - continuous observation environment benchmark, e.g. cartpole, from pixels (atari, craftax), robotic (simulated engine), 
  - stochastic environment benchmark where either observations are stochastic wrt to hidden state, or hidden states evolve stochastically or both.

World modeling benchmarks: instead of measuring the efficiency by which the agent optimizes the reward function, these test the quality of the world model (e.g. through prediction ability, generalization etc)
- AutumnBench
- Worldtest

We deliberately include two classes of benchmarks here, and it's possible that they might be exclusive in the sense that they would make sense to be pursued in different projects. The project will aim to identify the most promising directions across this space, before committing to either or both.


### Controlling for pretraining memorization
A criticism of Pinductor and related approaches is that the language model is recalling environments from pretraining, and therefore that the method is not generalizing. We address this through a mix of the following: the first two are important; the third point is optional.

- Benchmarking on environments that are absent from LLM pretraining. This is one of the key reasons for ARC-AGI-3.

- For each benchmark that could be present in LLM pre-training, there should be there should be two controlled semantic ablations to check the extent to which the language model is recalling from pretraining:
1. Change the names to other names.
2. Change the names to random strings.
For each of these ablations, we should report the percentage of times the LM still recognizes the semantically altered environment, e.g. by analyzing the percentage of times it mentions it in the reasoning trace (i.e., it identified the correct environment from memory, which is only weak generalization).

- OPTIONAL: Introduce a synthetic environment with no natural structure. In this case we expect that the LLM underperforms. If that weren't the case, then this violates the idea that the method has a prior for natural environments. This is a sanity check.


### Baselines

- The project will use Pinductor and other sample-efficient RL benchmarks as baseline, as well as the oracles / ground truth baselines as proposed in the initial Pinductor paper. 
- OPTIONAL: Time permitting, one could also benchmark with humans on the same environments using the same prompt, ie same set of instructions. Human subjects can be gathered through Amazon Mechanical Turk.


## Candidate improvements to Pinductor

The project will pick and choose between the following proposed improvements to Pinductor. These are classified as easy, medium and hard difficulty. These are of course non-exhaustive, and other improvements are encouraged. 


### Easy

#### Must have

- Update Pinductor for it to learn a standard POMDP (https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process#Formal_definition), which should facilitate learning compared to current setup. By this we mean two things:
  - Make the observation function not dependent on the previous action, just on the current state. This is standard and should greatly facilitate learning of observation model.
  - Return an observation from the initial state through the observation model before taking any action. This is also standard and should greatly facilitate learning of initial state models.

- Instead of returning the sum of expected likelihoods to the LLM as a signal, return the expected likelihood at each time --- this should greatly facilitate improving the model because it provides a richer signal --- but we would still use the sum of expected likelihoods as the final objective. 
  - As a next step, figure out the exact relationship between the sum of expected likelihoods and the ELBO, and make the final objective an actual ELBO. What might be missing to the current objective to make it an elbo is the KL divergence between Q and bar Q at each time step, where Q is the approximate posterior after re-weighting the particles and bar Q is the distribution before re-weighting, but after propagating the particles from the previous timestep with the transition function and the last action. Bar Q can hence be considered a prior over the state, before doing the Bayesian update, which is the re-weighting with observations to get Q. When you score the KL divergence between the prior and the posterior, you get the other term that's in the ELBO. If you consider that at each time step and make the sum, you get the exact same naive ELBO that's used in the FPI inference algorithm in the PyMDP library. Therefore, please use the FPI inference algorithm in the PyMDP library.

#### Nice to have

- Introducing LLM guided planning. E.g. this should help trim wrong A* branch and help the model to plan better by leveraging the prior of the LLM on the downstream task.

- Entertaining parallel models and include model information gain in planner. While this would enable the agent to seek data that helps to disambiguate between models, it will also greatly increase computational cost. Therefore, it's really nice to explore this, but we should take the added computational cost into account in our benchmarks, and we'll assess to what extent this added cost is worth it.

### Medium

#### Must have

- Improving planner and inference algorithm to improve performance *and* get rid of the compass assumption. ie using no priviledged information.

#### Nice to have

- Have the LLM guess the state space -> full causal representation learning

- The high variance of the pipeline is a problem. Can we somehow reduce this?

- Improving architecture so that online learning phase with the LLM is much stronger, and in particular so things could work well without offline learning.

- Introduce expected information gain about states and parameters in the planning algorithm. Because of the code structure of the models, information gain over states is done straightforward via particle filtering and alternatives, but information gain about parameters should be very hard and therefore less of a priority.

- Experiment beyond particle filtering: Variational Sequential Monte Carlo, variational particle approximations, and other ideas.

- Our pipeline has the promise to be truly useful on a state-of-the-art VLM instead of an LLM. Indeed, state-of-the-art robotics are using VLAs. If we can show we improve reasoning and planning using a VLM that generates code, then we are in for improving on the state of the art. Things should already work with these small models we're currently using, though, before we move on to the really big models. 

- Improving planner for exploiting good ELBO Models. At the moment, some good ELBO models get bad performance as measured by reward.


### Hard

- Generalization will always be a problem because LLMs are limited at it. This means that for the pipeline to be truly useful, we need to solve this online learning problem. Can we do this? This means solving the code repair problem with a better signal.

- Consider hierarchical space of POMDPs and use the LLM for proposing hierarchical models for multi-level planning, e.g. [ahmedSynthesizingWorldModels2025] 

- Mix and match between LLM and classical methods to learn the state space and the parameters of the model. Produce a rigorous baseline analysis, to see when the LLM helps, when it should be supplemented with another classical approach to estimation, and when the classical approach suffices.

- Fine tuning LLMs for proposing good models and refining the models.


## Useful references

The project will start by reviewing recent advances in automated program synthesis, active model discovery and ARC-AGI-3, to make sure we are in sync with the latest developments:

@misc{murphyModelDiscoveryAgent2026,
  title = {Model {{Discovery Agent}}: {{LLM-assisted Bayesian}} Experiment Design for Data-Efficient Discovery of Mechanistic World Models},
  shorttitle = {Model {{Discovery Agent}}},
  author = {Murphy, Kevin},
  year = 2026,
  month = aug,
  publisher = {arXiv},
  doi = {10.48550/ARXIV.2608.09696},
  urldate = {2026-08-12},
  abstract = {Predicting the answer to interventional ``what if'' questions --- the outcome of an action never taken --- requires a \textbackslash emph\textbraceleft mechanistic\textbraceright, causal model, not a curve fit; and learning such a model requires \textbackslash emph\textbraceleft experiments\textbraceright, because passive data leaves its mechanisms unidentified. Experiments are expensive, so the central problem is \textbackslash emph\textbraceleft data efficiency\textbraceright. We present the Model Discovery Agent (MDA), which couples a large language model (LLM), used as a \textbackslash emph\textbraceleft proposer\textbraceright{} of candidate structures, with standard Bayesian machinery --- sequential Monte Carlo (SMC) for parameter and structure posteriors, simulation-based inference (SBI) for intractable likelihoods, and value-of-information (VoI) for experiment design --- to discover latent mechanistic world models from few interventions. MDA operates in the M-open setting: when the truth lies outside the current hypothesis class, a predictive check flags the inadequacy and the proposer expands the hypothesis space with a new model whose parameters are then identified by designed experiments. We show that \textbackslash emph\textbraceleft discovery and design reinforce\textbraceright : the design step identifies the mechanism the discovery step proposes, and the identified mechanism improves predictions, enabling further discoveries from the remaining unexplained residuals. On three different benchmarks --- covering physics (\textbackslash DPbench, \textbackslash citep\textbraceleft wiemann2026discoverphysics\textbraceright ), chemistry (\textbackslash CHEMbench, \textbackslash citep\textbraceleft kabra2026autoscilab\textbraceright ) and biology (\textbackslash HHbench, a new partially observed single-neuron electrophysiology benchmark we create) --- we show that MDA sets a new SOTA in terms of data-efficient model learning and reliable interventional prediction ability.},
  copyright = {Creative Commons Attribution 4.0 International},
  keywords = {Artificial Intelligence (cs.AI),FOS: Computer and information sciences},
  file = {/Users/lancelotdacosta/Zotero/storage/W834VVJG/Murphy - 2026 - Model Discovery Agent LLM-assisted Bayesian experiment design for data-efficient discovery of mecha.pdf}
}


@misc{ahmedSynthesizingWorldModels2025,
  ids = {ahmedSynthesizingWorldModels2025a},
  title = {Synthesizing World Models for Bilevel Planning},
  author = {Ahmed, Zergham and Tenenbaum, Joshua B. and Bates, Christopher J. and Gershman, Samuel J.},
  year = 2025,
  month = jul,
  number = {arXiv:2503.20124},
  eprint = {2503.20124},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2503.20124},
  urldate = {2025-12-12},
  abstract = {Modern reinforcement learning (RL) systems have demonstrated remarkable capabilities in complex environments, such as video games. However, they still fall short of achieving human-like sample efficiency and adaptability when learning new domains. Theory-based reinforcement learning (TBRL) is an algorithmic framework specifically designed to address this gap. Modeled on cognitive theories, TBRL leverages structured, causal world models - "theories" - as forward simulators for use in planning, generalization and exploration. Although current TBRL systems provide compelling explanations of how humans learn to play video games, they face several technical limitations: their theory languages are restrictive, and their planning algorithms are not scalable. To address these challenges, we introduce TheoryCoder, an instantiation of TBRL that exploits hierarchical representations of theories and efficient program synthesis methods for more powerful learning and planning. TheoryCoder equips agents with general-purpose abstractions (e.g., "move to"), which are then grounded in a particular environment by learning a low-level transition model (a Python program synthesized from observations by a large language model). A bilevel planning algorithm can exploit this hierarchical structure to solve large domains. We demonstrate that this approach can be successfully applied to diverse and challenging grid-world games, where approaches based on directly synthesizing a policy perform poorly. Ablation studies demonstrate the benefits of using hierarchical abstractions.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Artificial Intelligence},
  file = {/Users/lancelotdacosta/Zotero/storage/HSSR62J3/Ahmed et al. - 2025 - Synthesizing world models for bilevel planning.pdf;/Users/lancelotdacosta/Zotero/storage/MNG4RCIE/2503.html}
}


@misc{astrosegerArc3AgentsBaseline,
  title = {Arc-3-Agents-Baseline1: Game-General Coding Agents for {{ARC-AGI-3}}},
  author = {{astroseger}},
  howpublished = {\url{https://github.com/astroseger/arc-3-agents-baseline1}},
  urldate = {2026-08-19},
  note = {Experimental coding agents that solve ARC-AGI-3 games with no game-specific code or information --- the same agent code and prompts are used across games. The ewma\_sv\_v1.6 agent maintains an executable world model and verifies observations through replay, reportedly solving all 25 public games at ~99\% of human efficiency; twma\_v1.6 is a simpler textual world model with lower compute overhead. Directly relevant to the executable-world-model and code-repair questions in this project, and a natural baseline on ARC-AGI-3.}
}


@misc{schemaHarnessFrontierModels2026,
  title = {Frontier Models with Our Harness Achieve ~99\% on {{ARC-AGI-3}} Public},
  author = {{Schema}},
  howpublished = {\url{https://schema-harness.github.io}},
  urldate = {2026-08-19},
  note = {Results page reporting ~99\% on the 25 public ARC-AGI-3 games using a harness around frontier models. Relevant as a strong ARC-AGI-3 baseline and as evidence on how much of the benchmark is addressable by scaffolding alone --- worth checking what sample and compute budget these numbers assume before using them as a comparison point.}
}








