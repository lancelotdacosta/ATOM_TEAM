# Benchmarking ATOM agents

For context, ATOM is developing an agent foundation model: A machine learning model capable of (1) interaction (action, planning), (2) sample-efficient, continual and open-ended learning, (3) generalization and adaptation under distribution shifts, (4) alignment, safety and robustness.
The current program makes certain commitments about the types of models and agents to be developed: https://blog.lancelotdacosta.com/posts/2026-07-06-atom-research-roadmap/ (please read this!). 
In a nutshell, it's about:
- Having uncertainty quantification, which leads to information gain-driven actions, which leads to sample efficiency. Importantly uncertainty quantification should be done in a way that the computational cost doesn't blow up. 
- Additional commitments to:
  - hierarchical models are expected to lower model complexity, which should lead to further improvements in sample and compute efficiency, as well as improved generalization out of distribution.
  - structured models are expected to lead to improved generalization out of distribution due to lower complexity and causality arguments. 

The question this project asks is: **how do we benchmark the agents developed by ATOM? What are these agents going to be good at, and what is a meaningful signal from which we can discern a positive architectural change versus noise?**

## A first step: benchmarking sample and compute efficient RL 

A first step and low-hanging fruit is to focus on the idea of sample and compute efficient reinforcement learning. Researchers are becoming increasingly aware that sample and compute efficiency should be accounted for in the benchmarks. Before, these numbers were loosely reported but not part of the benchmark. One of the benchmarks that promoted this and integrated this by design is ARC-AGI, where you have a bounded access to compute and you have to perform the task in only a few samples. We believe that sample and compute should be benchmarked as rigorously and systematically as performance.

This project will first be a pilot study about what has been done in benchmarking sample and compute-efficient RL and what are the main baselines. For baselines in particular, IRIS and Dreamer are good starting points. If this is still open problem, we will produce a benchmark of RL environments where every agent run is plotted in an x-y-z axis: where x is compute, y is samples, and z is reward. Note that using existing RL environments works.

Then we need to combine discrete environments, continuous environments. The pre-question is: how do you systematically evaluate compute across different architectures on different hardware for discrete environments? For continuous environments, reward could potentially be replaced by time in the game, and we could also think of replacing compute cost with latency, but that's TBD.

Actually, advers have low latency but high inference cost. So maybe we just keep the compute cost. Importantly, RAGI just assesses models that are already trained and then do few-shot adaptation or in-context learning. These are models that already have good priors.

Here, we also care about assessing the ability of learning those priors, which means that we want the learning curve from not being competent at all in an environment to being fully competent, and how competency evolves as a number of samples and compute. This competency could be measured by the reward at every environment, at every episode, I mean.

To get a final number, every agent could be ranked according to learning divided by samples times compute. Learning could be measured by the difference between reward in the first episode(s) and reward in the final episode(s).

The goal is to have systematic and rigorous assessment in which you compare the different baselines, to have a wide variety of environments, to have a website where we can show how each baseline performs according to this axis and compare. What would also be nice, but that's optional, is to get a baseline which is human performance on these games, which could be acquired through Mechanical Turk, for example. 




