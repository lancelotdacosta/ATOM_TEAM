# The natural building block approach to hierarchical world models
Author: Lance, Date: 2026-09-23

## Motivation

This project explores natural building blocks in the context of hierarchical models. In a previous paper, Natural Building Blocks for Structured World Models [costaNaturalBuildingBlocks2025], we outlined a vision for how world models could be created. The idea is the following.

First, consider the analogy of deep learning. Deep learning initially is just about approximating functions. Deep learning says that you just need a small class of base layers, and to be able to backprop through them. Once you're able to do that, the next thing you have to figure out is how to backprop through a stack of layers. If you solve it, then you can approximate arbitrary functions. So the problem of function approximation, which is itself very hard, is just decomposed into having fundamental building blocks for which you can do your calculus and operate, and being able to stack those building blocks to obtain systems of increasing complexity. This is it: base layers and composition. The natural building blocks approach explores the same idea in the context of world models.

A world model is an approximation of the world. The world is generally considered to be a complex stochastic process: a dynamical system that may have some degree of randomness to it. In the simplest case, think of the motion of rigid bodies under Newtonian laws. But then once you introduce friction, the best model may become partly random, because you can't usually model the friction or other fluctuations well. In statistical mechanics, when you model the motion of molecules, models include a random term because of the thermal fluctuations of the molecules. In scientific modeling, random terms are usually used because things are noisy. Point being, the world is well modelled as a stochastic process.

We would like to be able to approximate all kinds of worlds that we might encounter. Being able to do so allows us to plan inside of our world model, simulate counterfactual or hypothetical courses of action, assess their consequences, refine, imagine alternative futures, and so on. Everything that a world model is good for. And we want to address this problem of world modeling in the same way that made deep learning so successful: a small set of base layers and a way to stack them on top of each other so that things still operate.

What's the best way of doing so? The most direct way is to consider as building blocks the most common stochastic processes that are encountered in the sciences. These are discrete-state Markov chains and continuous-state diffusion processes. (Note that here a diffusion process is not necessarily meant in the way of a diffusion model. These are related, but here a diffusion process just means a stochastic differential equation driven by white noise. A diffusion model can be used to parameterize or learn such a stochastic differential equation, but there are other ways as well. We'll come back to this later.)

A lot of the way we think about the world and model the world is discrete. Think of: this is hot, this is cold, this is lukewarm. This is a discrete enumeration of possibilities. Or: go left, go right, go straight. This is a discrete enumeration of possible directions to go in. Markov chains are the simplest stochastic process describing motion when the states of the world are discrete.

We do also represent a lot of things with continuous variables. What's the temperature tomorrow? Say around 25 degrees Celsius. This is a continuous representation, and the simplest way of encoding motion between continuous representations is with a diffusion process.

The natural building block approach asks: can we stack hierarchies of Markov chains and diffusion processes to approximate worlds? How expressive are the resulting hierarchies? What can we not approximate in this way? And so on and so forth.

## The two natural building blocks

The first step would be to read the aforementioned paper, Natural Building Blocks for Structured World Models [costaNaturalBuildingBlocks2025]. There are a couple of things I didn't mention in the motivation, which is that for a world model to be useful to an agent, it needs to incorporate action variables as well.

So we're not merely looking for Markov chains and diffusions, but action-conditioned Markov chains, which are essentially Markov decision processes without a reward function. And we're looking for action-conditioned diffusions, which are a class of controlled stochastic differential equations. The world model of an agent would then be a hierarchy of controlled and uncontrolled Markov chains and controlled and uncontrolled diffusions.

Some would be uncontrolled, of course, because there are some states that cannot be controlled by the agent, but may still be useful to model. For example, if they can be controlled indirectly through other states, but not directly.

Another crucial component lacking from the motivation above is the case of partial observability. Most things in the world cannot be observed directly, but must instead be inferred from data. This means that the fundamental building block would be a partially observed controlled Markov chain and a partially observed controlled diffusion. These are instances of partially observed Markov decision processes (POMDPs) with an assumption on what constitutes the latent dynamics (Markov chains or diffusions).

So here are two building blocks: POMDPs driven by Markov chains and POMDPs driven by diffusion processes. The hypothesis is that being able to work with these two single building blocks and stacking them on top of each other actually enables a very expressive class of hierarchical world models. The main reference [costaNaturalBuildingBlocks2025] shows evidence for why this is true, reviewing literature showing video and sound generative modeling, agents playing simple games that are simpler than Atari, and simulated robotic arm control.

Before we move on to describing the project itself, let's answer why we need continuous and discrete states. There's no consensus in the machine learning literature, but in neuroscience, there is evidence or at least a belief that the brain uses continuous representations for everything that's low level, like audition, vision, and so on, and discrete representations for everything that's high level and abstract, such as high-level reasoning. The motivation is that combining these two types of representations will get us much further than either alone.

## Project goal

**TL;DR: Build an agent with a hierarchical world model made from one natural building block (discrete POMDPs are the easiest start), learn its parameters from data, and show it is competitive with PPO-GRU, DreamerV3, and Pinductor on MiniGrid in reward per sample and per unit of compute.**

The goal of the project is to produce a working hierarchical generative model agent using one of these core building blocks. What's important is that the parameters are not fixed, but are learned from data. The hierarchical structure, however, may be fixed or learned from data, cf. [dacostaPossiblePrinciplesAligned2024, doshi-velezInfinitePartiallyObservable2009], depending on what works best. Note, the fundamental building block that's discrete is much more mature (i.e. the discrete POMDP), so that may be the easiest place to start experimenting.

The goal is then to deploy the architecture in a synthetic RL environment like MiniGrid and show competitive performance with respect to alternatives. The choice of MiniGrid should be made if the choice of natural building block is discrete, but other simple environments for continuous variables should be used if one explores the continuous building block. The easiest to start experimenting with on that front would be CartPole, for example.

Performance should be assessed by looking at reward or learning performance per amount of samples ingested and per amount of compute. So the benchmarked methods should be evaluated on 3D axes. Alternatives that need benchmarking include state-of-the-art model-free RL approaches like PPO-GRU, and model-based approaches like DreamerV3 [hafnerMasteringDiverseDomains2023] and Pinductor [sixLearningPOMDPWorld2026]. Pinductor is another one of ATOM's agents.

Throughout, the agent should maintain uncertainty about some things. I advocate for the states and states only to start with. Please see this section on tips later. The goal is to use this uncertainty in decision making through information gain. This is expected to yield increased sample efficiency.

## Future goals

After this initial milestone is reached (note that this initial milestone could constitute at least a workshop paper, depending on the results), one could examine hierarchies with the other building block, and then hierarchies mixing the two. For now, we expect that when mixing both types of layers, it's best to use discrete layers at the top and continuous layers at the bottom to have these low-level continuous, high-level discrete hierarchies. And it's not clear how one would even do the opposite mathematically.

## Steps (discrete natural building block)

The first step would be to become familiar with the first natural building block, the discrete layer, or the second natural building block, the continuous layer. For the discrete layer, a good place to start experimenting is the pymdp codebase [heinsPymdpPythonLibrary2022]. Essentially, there are many ways of working with the discrete POMDP, many ways one can do inference about states and learn the parameters, and it's generally hard. So it's good to experiment with these things for a while and get a feel for them. It's good not to zoom in too much on pymdp because it's only one specific approach. Please look into other papers using discrete POMDPs such as the Bayes-adaptive POMDP paper [rossBayesAdaptivePOMDPs2008a, rossBayesianApproachLearning2011, kattBayesianReinforcementLearning2018]. And look around for more to get a feel of the landscape. What would be important is to experiment with different kinds of POMDPs as well, like POMDPs where the latent is a transformer, for example. So don't take too much for granted that the right base building block is a Markov chain POMDP. It could be something more complex. What's important would be to focus on one type of representation, here discrete, and compare all the model types and hierarchies. Depending on the latent dynamics which are chosen, as long as they have discrete states, there would be different properties. So it's good to benchmark and explore these different architectures. More on this in the design choices and challenges section. Note that if one were to start with the continuous building block, a natural reference would be AXIOM [heinsAXIOMLearningPlay2025].

Once you're familiar with discrete POMDPs and have a method for learning parameters (it doesn't need to be extremely good), it's now time to move on to the hierarchy. Here, good references are the RGM papers [fristonPixelsPlanningScalefree2024, Fig. 3; zaghwRenormalisingGenerativeModels2026] and the Thomas Parr papers [parrDiscreteContinuousBrain2018a, parrComputationalNeurologyMovement2021]. They show two ways of doing hierarchies, and please do not take them too much to the letter. In the latter two, a state at the higher level can be an initial condition for the trajectory at the lower level, in the former a state at the higher level can condition the whole trajectory at the lower level. It's important to try both, and the state being an initial condition for the trajectory at the lower level feels like it would be richer and potentially better.

## Design choices and challenges

There are two main design choices here. The first is the exact parametrization of the natural building block. Either you take your latent process to be a Markov chain, which is a traditional POMDP, or you make it a transformer, or you look into other architectures that people have tried. This is very important to look into and benchmark.

The second main design choice is how you do the hierarchy. For example, do states at the higher level correspond to trajectories at the lower level, or are states at the higher level initial conditions for the trajectories at the lower level? I'm sure people have tried other things as well. How does that work?

Beyond these design choices, which will lead to many different kinds of things to try, the main question is how to enable parameter learning to work in these hierarchies, knowing that good parameter learning is already very challenging in a one-layer POMDP. This is because the transition distribution and the observation distribution are heavily entangled.

## Some tips

For every unknown, whether it's a state or a parameter, you have the option of inferring a point estimate, e.g. regularized maximum likelihood, or inferring a distribution around it, i.e. Bayesian inference. Inferring a distribution around it means that you have uncertainty, usually at the expense of more computational cost, and this uncertainty can be leveraged in decision making to make your agent more sample efficient.

Getting good parameter learning with distributions, i.e. Bayesian parameter learning, is very challenging. So it would be much easier to start with just point estimates over the parameters. Bayesian parameter learning is a whole different beast, so please leave it aside in the first instance.

## Useful reading

For more context on the program and inference and learning in hierarchical models, the vision which we're aiming for and so on, please refer to the ATOM program [dacostaActiveWorldModeling2026] and the foundational paper, Possible principles for aligned structure learning agents [dacostaPossiblePrinciplesAligned2024].

## Resources

Within ATOM, experts on discrete POMDPs with latent Markov chains include Andrew Pashea, Wouter Nuijten, and Marc Pritsch. Feel free to reach out to them for questions.

It would be important to work in sync with ATOM's principled hierarchies project so that the emerging theory informs the developments, and so that the empirical developments inform the theory.

## References

```bibtex
@misc{dacostaActiveWorldModeling2026,
  title = {Toward {{Active World Modeling}}},
  author = {Da Costa, Lancelot},
  year = 2026,
  month = jul,
  howpublished = {\url{https://blog.lancelotdacosta.com/posts/2026-07-06-atom-research-roadmap/}},
  note = {ATOM research roadmap, blog post}
}

@misc{costaNaturalBuildingBlocks2025,
  title = {Natural {{Building Blocks}} for {{Structured World Models}}: {{Theory}}, {{Evidence}}, and {{Scaling}}},
  author = {Da Costa, Lancelot and Namjoshi, Sanjeev and Ansari, Mohammed Abbas and Sch{\"o}lkopf, Bernhard},
  year = 2025,
  month = nov,
  number = {arXiv:2511.02091},
  eprint = {2511.02091},
  archiveprefix = {arXiv},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2511.02091}
}

@misc{dacostaPossiblePrinciplesAligned2024,
  title = {Possible Principles for Aligned Structure Learning Agents},
  author = {Da Costa, Lancelot and Gaven{\v c}iak, Tom{\'a}{\v s} and Hyland, David and Samiei, Mandana and {Dragos-Manta}, Cristian and Pattisapu, Candice and Razi, Adeel and Friston, Karl},
  year = 2024,
  month = sep,
  number = {arXiv:2410.00258},
  eprint = {2410.00258},
  archiveprefix = {arXiv},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2410.00258}
}

@inproceedings{doshi-velezInfinitePartiallyObservable2009,
  title = {The {{Infinite Partially Observable Markov Decision Process}}},
  booktitle = {Advances in {{Neural Information Processing Systems}}},
  author = {{Doshi-Velez}, Finale},
  year = 2009,
  volume = {22},
  publisher = {Curran Associates, Inc.}
}

@misc{fristonPixelsPlanningScalefree2024,
  title = {From Pixels to Planning: Scale-Free Active Inference},
  author = {Friston, Karl and Heins, Conor and Verbelen, Tim and Da Costa, Lancelot and Salvatori, Tommaso and Markovic, Dimitrije and Tschantz, Alexander and Koudahl, Magnus and Buckley, Christopher and Parr, Thomas},
  year = 2024,
  month = jul,
  number = {arXiv:2407.20292},
  eprint = {2407.20292},
  archiveprefix = {arXiv},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2407.20292}
}

@misc{hafnerMasteringDiverseDomains2023,
  title = {Mastering {{Diverse Domains}} through {{World Models}}},
  author = {Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  year = 2023,
  month = jan,
  number = {arXiv:2301.04104},
  eprint = {2301.04104},
  archiveprefix = {arXiv},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2301.04104}
}

@misc{heinsAXIOMLearningPlay2025,
  title = {{{AXIOM}}: {{Learning}} to {{Play Games}} in {{Minutes}} with {{Expanding Object-Centric Models}}},
  author = {Heins, Conor and Van de Maele, Toon and Tschantz, Alexander and Linander, Hampus and Markovic, Dimitrije and Salvatori, Tommaso and Pezzato, Corrado and {\c C}atal, Ozan and Wei, Ran and Koudahl, Magnus and Perin, Marco and Friston, Karl and Verbelen, Tim and Buckley, Christopher},
  year = 2025,
  month = may,
  number = {arXiv:2505.24784},
  eprint = {2505.24784},
  archiveprefix = {arXiv},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2505.24784}
}

@misc{heinsPymdpPythonLibrary2022,
  title = {Pymdp: {{A Python}} Library for Active Inference in Discrete State Spaces},
  author = {Heins, Conor and Millidge, Beren and Demekas, Daphne and Klein, Brennan and Friston, Karl and Couzin, Iain and Tschantz, Alexander},
  year = 2022,
  month = jan,
  number = {arXiv:2201.03904},
  eprint = {2201.03904},
  archiveprefix = {arXiv},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2201.03904}
}

@misc{kattBayesianReinforcementLearning2018,
  title = {Bayesian {{Reinforcement Learning}} in {{Factored POMDPs}}},
  author = {Katt, Sammie and Oliehoek, Frans and Amato, Christopher},
  year = 2018,
  month = nov,
  number = {arXiv:1811.05612},
  eprint = {1811.05612},
  archiveprefix = {arXiv},
  publisher = {arXiv},
  doi = {10.48550/arXiv.1811.05612}
}

@article{parrComputationalNeurologyMovement2021,
  title = {The Computational Neurology of Movement under Active Inference},
  author = {Parr, Thomas and Limanowski, Jakub and Rawji, Vishal and Friston, Karl},
  year = 2021,
  month = jun,
  journal = {Brain},
  volume = {144},
  number = {6},
  pages = {1799--1818},
  doi = {10.1093/brain/awab085}
}

@article{parrDiscreteContinuousBrain2018a,
  title = {The {{Discrete}} and {{Continuous Brain}}: {{From Decisions}} to {{Movement}}---{{And Back Again}}},
  author = {Parr, Thomas and Friston, Karl J.},
  year = 2018,
  month = sep,
  journal = {Neural Computation},
  volume = {30},
  number = {9},
  pages = {2319--2347},
  doi = {10.1162/neco_a_01102}
}

@inproceedings{rossBayesAdaptivePOMDPs2008a,
  title = {Bayes-{{Adaptive POMDPs}}},
  booktitle = {Advances in {{Neural Information Processing Systems}}},
  author = {Ross, St{\'e}phane and {Chaib-draa}, Brahim and Pineau, Joelle},
  year = 2008,
  volume = {20},
  pages = {1225--1232},
  publisher = {Curran Associates, Inc.}
}

@article{rossBayesianApproachLearning2011,
  title = {A {{Bayesian Approach}} for {{Learning}} and {{Planning}} in {{Partially Observable Markov Decision Processes}}},
  author = {Ross, St{\'e}phane and Pineau, Joelle and {Chaib-draa}, Brahim and Kreitmann, Pierre},
  year = 2011,
  journal = {Journal of Machine Learning Research},
  volume = {12},
  pages = {1729--1770}
}

@misc{sixLearningPOMDPWorld2026,
  title = {Learning {{POMDP World Models}} from {{Observations}} with {{Language-Model Priors}}},
  author = {Six, Valentin and Panse, Frederik and Fajeau, Mathis and Da Costa, Lancelot and Sharma, Mridul and Amayuelas, Alfonso and Xiao, Tim Z. and Hyland, David and Hennig, Philipp and Sch{\"o}lkopf, Bernhard},
  year = 2026,
  month = may,
  number = {arXiv:2605.13740},
  eprint = {2605.13740},
  archiveprefix = {arXiv},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2605.13740}
}

@misc{zaghwRenormalisingGenerativeModels2026,
  title = {Renormalising {{Generative Models}} for {{Active Inference}}: {{Foundations}}, {{Derivations}}, and {{Verification}}},
  author = {Zaghw, Karim and Pashea, Andrew and Pritsch, Marc and Nuijten, Wouter and Friston, Karl and Da Costa, Lancelot},
  year = 2026,
  month = aug,
  number = {arXiv:2608.09512},
  eprint = {2608.09512},
  archiveprefix = {arXiv},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2608.09512}
}
```
