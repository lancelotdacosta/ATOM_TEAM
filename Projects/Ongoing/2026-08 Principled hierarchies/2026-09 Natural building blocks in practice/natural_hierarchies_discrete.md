# The natural building block approach to hierarchical world models

## Motivation

This project explores natural building blocks in the context of hierarchical models. In a previous paper, Natural Building Blocks for Structure World Models, we outlined a vision for how world models could be created. The idea is the following. First, consider the analogy of deep learning.

Deep learning initially is just about approximating functions. Deep learning says that you just need a small class of base layers, and to be able to backprop through them. Once you're able to do that, the next thing you have to figure out is how to backprop through a stack of layers.

If you solve it, then you can approximate arbitrary functions. So the problem of function approximation, which is itself very hard, is just decomposed into having fundamental building blocks for which you can do your calculus and operate, and being able to stack those building blocks to obtain systems of increasing complexity.

This is it: base layers and composition. The natural building blocks approach explores the same idea in the context of world models. A world model is an approximation of the world. The world is generally considered to be a complex stochastic process.

This is a dynamic that may have some degree of randomness to it. In the simplest case, think of the motion of rigid bodies under Newtonian laws. But then once you introduce friction, maybe the best model... Because a deterministic becomes partly random because you can't usually model the friction or other fluctuations well.

In statistical mechanics, when you model the motion of molecules, models include a random term because of the thermal fluctuations of the molecules. In science, random terms are usually used because things are noisy. Point being, the world is a stochastic process. Now, we would like to be able to approximate all kinds of worlds that we might encounter.

Being able to do so allows us to plan inside of our world model, simulate counterfactual or hypothetical courses of action, assess their consequences, refine, imagine alternative futures, and so on. Everything that a world model is good for. And we want to address this problem of world modeling in the same way that deep learning, that made deep learning so successful: a small set of base layers and a way to stack them on top of each other so that things still operate. 

What's the best way of doing so? The most direct way is to consider as building blocks the most common stochastic processes that are encountered in the science. These are discrete-state Markov chains and continuous-state diffusion processes. (Note that here a diffusion process is not necessarily meant in the way of a diffusion model. These are related, but here a diffusion process just means a stochastic differential equation driven by white noise. A diffusion process can be used to parameterize or learn such a stochastic differential equation, but there are other ways as well. We'll come back to this later). Markov chains are the simplest stochastic process describing motion when the states of the world are discrete. Think of this is hot, this is cold, this is lukewarm. This is a discrete enumeration of possibilities, or go left, go right, go straight. This is a discrete enumeration of possible directions to go into.

A lot of the way we think about the world and model the world is discrete. And the simplest way of representing motion on a discrete space is through a Markov chain. But we do also represent a lot of things with continuous variables. What's the temperature tomorrow? Say around 25 degrees Celsius. This is a continuous representation, and the simplest way of encoding motion between continuous representations is with a diffusion process. The natural building block approach asks, can we stack hierarchies of Markov chains and diffusion processes to approximate worlds? How expressive are the resulting hierarchies? What can we not approximate in this way? And so on and so forth. 

## The two natural building blocks

The first step would be to read the aforementioned paper, Natural Building Blocks for Structural World Models. There are a couple of things I didn't mention in the motivation, which is that for a world model to be useful to an agent, it needs to incorporate action variables as well.

So we're not merely looking for a Markov chains and diffusions, but action-conditioned Markov chains, which are essentially Markov decision processes without a reward function. And we're looking for action-conditioned diffusions, which are a class of controlled stochastic differential equations. The world model of an agent would then be a hierarchy of controlled and uncontrolled Markov chains and controlled and uncontrolled diffusions.

Some would be uncontrolled, of course, because there are some states that cannot be controlled by the agent, but may still be useful to model. For example, if they can be controlled indirectly through other states, but not directly. 

Another crucial component lacking from the motivation above is the case of partial observability. Most things in the world cannot be observed directly, but must instead be inferred from data. This means that the fundamental building block would be a partially observed controlled Markov chain and a partially observed controlled diffusion. These are instances of partially observed Markov decision processes (POMDPs) with an assumption on what constitutes the latent dynamics.

So here are two building blocks: POMDPs driven by Markov chains and POMDPs driven by diffusion processes. The hypothesis is that being able to work with these two single building blocks and stacking them on top of each other actually enables a very expressive class of hierarchical world models. The main reference shows evidence for why this is true, showing video and sound generative modeling agents playing simple games that are simpler than Atari or simulated robotic arm control.

Before we move on to describing the project itself, let's answer why do we need continuous and discrete states. There's no consensus in the machine learning literature, but in neuroscience, there is evidence or at least a belief that the brain uses continuous representations for everything that's low level, like audition, vision, and so on, and discrete representations for everything that's high level and abstract, such as high-level reasoning. We think that combining these two types of representations, inspired by this, will get us much further than either alone.

## Project goal

The goal of the project is to produce a working hierarchical generative model using one of these core building blocks. What's important is that the parameters are not fixed, but are learned from data. The hierarchical structure, however, may be fixed or learned from data, cf [possible principles, infinite pomdp], depending on what works best. Note, the fundamental building block that's discrete is much more mature (i.e. discrete pomdp), so that may be the easiest place to start experimenting.

The goal is then to deploy the architecture in a synthetic RL environment like MiniGrid and show competitive performance with respect to alternatives. The choice of MiniGrid should be if the choice of natural building block is discrete, but other simple environments for continuous variables should be used if one explores the continuous building block. The easiest to start experimenting on that front would be CartPole, for example.

Performance should be assessed by looking at reward or learning performance per amount of samples ingested and per amount of compute. So our methods benchmarked should be evaluated on a 3D axis. Alternatives that need benchmarking include state-of-the-art model-free RL approaches like PPO, GRU, Dreamer V3, and Pinductor. Pinductor is another one of Atom's agents.

## Future goals

After this initial milestone is solved (note that this initial milestone could constitute at least a workshop paper, depending on the results). After this milestone is reached, one could examine hierarchies with the other building block, and then hierarchies mixing the two. For now, we expect that when mixing, it's best to use discrete layers at the top and continuous layers at the bottom to have this low-level continuous, high-level discrete hierarchies. And it's not clear to me how one would even do the opposite mathematically.

## Steps (discrete natural building block)

The first step would be to become familiar with the first natural building block, the discrete layer, or second natural building block, the continuous layer. For discrete layer, a good place to start experimenting is on the PyMDP codebase. Essentially, there are many ways of working with the discrete POMDP, many ways one can do inference about states and learn the parameters, and it's generally hard. So it's good to experiment with these things for a while and get a feel for them. It's good not to zoom in too much on PMDP because it's only one specific approach. Please look into other papers using discrete POMDPs such as the Bayes adaptive POMDP paper. And look around for more to get a feel of the landscape. What would be important is to experiment with different kinds of POMDPs as well, like POMDPs where the latent is a transformer, for example. Depending on the latent dynamic which is chosen, as long as it's discrete states, there would be different properties. So it's good to benchmark and explore these different architectures. More on this in the design choices and challenges section. Note that if one were to start with the discrete, the continuous building block, a natural reference would be AXIOM.

Once you're familiar with discrete POMDPs. And we have a method for learning parameters. It doesn't need to be extremely good. It's now time to move on to the hierarchy. Here, good references are the RGM paper [original, karim's paper] and the thomas parr paper. They show two ways of doing hierarchies, and please do not take them too much at the letter. But the point is, a state at the higher level can be an initial condition for the trajectory at the lower level, or a state at the higher level can condition the whole trajectory at the lower level. It's important to try both, and the state being an initial condition to the trajectory at the lower level feels like it would be richer and potentially better.

## Design choices and challenges

There's two main design choices here: the exact parametrization of the natural building block. There's two main ones. Either you take your latent process to be a Markov chain, which is a traditional POMDP, or you make it a transformer, or you look into other architectures that people have tried. 

The second main design choice is how you do the hierarchy. For example, are states at the higher level corresponding to trajectories at the lower level, or are states at the higher level initial conditions for the trajectories at the lower level? I'm sure people have tried other things as well. How does that work? 

Beyond these design choices, which will lead to many different kinds of things to try, the main question is how do you enable parametered learning to work in these hierarchies, knowing that good parametered learning already in a one-layer POMDP is very challenging. This is because the transition distribution and the observation distribution are heavily entangled.

## Some tips

For every unknown, whether it's a state or a parameter, you have the option of inferring a point estimate, e.g. regularized maximum likelihood, or inferring a distribution around it, ie Bayesian inference. Inferring a distribution around it means that you have uncertainty at the expense of computational cost, usually, and this uncertainty can be leveraged in decision making to make your agent more sample efficient. 

Getting good parameter learning with distributions, i.e. Bayesian parameter learning, is very challenging. So it would be very nice. It would be much easier to start with just point estimates over the parameters. Bayesian parameter learning is a whole different beast, so please leave aside in the first instance. 


## Useful reading

For more context on the program and inference and learning in hierarchical models, the vision which we're aiming for and so on, please refer to the ATOM program and the foundational paper. https://blog.lancelotdacosta.com/posts/2026-07-06-atom-research-roadmap/ and Possible principles for aligned structure learning agents.


## Resources

Within atom experts on discrete POMDPs with latent Markov chains include Andrew Pashea, Wouter Nuijten, Marc Pritsch. Feel free to reach out to them for questions. 

It would be important to work in sync with the atom's principled hierarchies project so that the emerging theory informs the developments, and so that the empirical developments inform the theory. 


## References

1. Natural Building Blocks for Structured World Models, the "previous paper" and "main reference" (lines 5, 25, 33). The title is written as "Structure World Models" on line 5 and "Structural World Models" on line 25; you may want to make those match.

@misc{costaNaturalBuildingBlocks2025,
  title = {Natural {{Building Blocks}} for {{Structured World Models}}: {{Theory}}, {{Evidence}}, and {{Scaling}}},
  shorttitle = {Natural {{Building Blocks}} for {{Structured World Models}}},
  author = {Da Costa, Lancelot and Namjoshi, Sanjeev and Ansari, Mohammed Abbas and Sch{\"o}lkopf, Bernhard},
  year = 2025,
  month = nov,
  number = {arXiv:2511.02091},
  eprint = {2511.02091},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2511.02091},
  urldate = {2025-11-05},
  abstract = {The field of world modeling is fragmented, with researchers developing bespoke architectures that rarely build upon each other. We propose a framework that specifies the natural building blocks for structured world models based on the fundamental stochastic processes that any world model must capture: discrete processes (logic, symbols) and continuous processes (physics, dynamics); the world model is then defined by the hierarchical composition of these building blocks. We examine Hidden Markov Models (HMMs) and switching linear dynamical systems (sLDS) as natural building blocks for discrete and continuous modeling--which become partially-observable Markov decision processes (POMDPs) and controlled sLDS when augmented with actions. This modular approach supports both passive modeling (generation, forecasting) and active control (planning, decision-making) within the same architecture. We avoid the combinatorial explosion of traditional structure learning by largely fixing the causal architecture and searching over only four depth parameters. We review practical expressiveness through multimodal generative modeling (passive) and planning from pixels (active), with performance competitive to neural approaches while maintaining interpretability. The core outstanding challenge is scalable joint structure-parameter learning; current methods finesse this by cleverly growing structure and parameters incrementally, but are limited in their scalability. If solved, these natural building blocks could provide foundational infrastructure for world modeling, analogous to how standardized layers enabled progress in deep learning.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Artificial Intelligence,Computer Science - Machine Learning},
  file = {/Users/lancelotdacosta/Zotero/storage/WUHHNJD3/Da Costa et al. - 2025 - Natural Building Blocks for Structured World Models Theory, Evidence, and Scaling.pdf;/Users/lancelotdacosta/Zotero/storage/HTKKI9IL/2511.html}
}


1. DreamerV3 (line 43)

@misc{hafnerMasteringDiverseDomains2023,
  title = {Mastering {{Diverse Domains}} through {{World Models}}},
  author = {Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  year = 2023,
  month = jan,
  publisher = {arXiv},
  urldate = {2023-01-17},
  abstract = {General intelligence requires solving tasks across many domains. Current reinforcement learning algorithms carry this potential but are held back by the resources and knowledge required to tune them for new tasks. We present DreamerV3, a general and scalable algorithm based on world models that outperforms previous approaches across a wide range of domains with fixed hyperparameters. These domains include continuous and discrete actions, visual and low-dimensional inputs, 2D and 3D worlds, different data budgets, reward frequencies, and reward scales. We observe favorable scaling properties of DreamerV3, with larger models directly translating to higher data-efficiency and final performance. Applied out of the box, DreamerV3 is the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula, a long-standing challenge in artificial intelligence. Our general algorithm makes reinforcement learning broadly applicable and allows scaling to hard decision-making problems.},
  keywords = {Computer Science - Artificial Intelligence,Computer Science - Machine Learning,Statistics - Machine Learning}
}


2. Pinductor, ATOM's agent (line 43)

@misc{sixLearningPOMDPWorld2026,
  title = {Learning {{POMDP World Models}} from {{Observations}} with {{Language-Model Priors}}},
  author = {Six, Valentin and Panse, Frederik and Fajeau, Mathis and Costa, Lancelot Da and Sharma, Mridul and Amayuelas, Alfonso and Xiao, Tim Z. and Hyland, David and Hennig, Philipp and Sch{\"o}lkopf, Bernhard},
  year = 2026,
  month = may,
  number = {arXiv:2605.13740},
  eprint = {2605.13740},
  primaryclass = {cs.LG},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2605.13740},
  urldate = {2026-09-23},
  abstract = {Whether navigating a building, operating a robot, or playing a game, an agent that acts effectively in an environment must first learn an internal model of how that environment works. Partially-observable Markov decision processes (POMDPs) provide a flexible modeling class for such internal world models, but learning them from observation-action trajectories alone is challenging and typically requires extensive environment interaction. We ask whether language-model priors can reduce costly interaction by leveraging prior knowledge, and introduce \textbackslash emph\textbraceleft Pinductor\textbraceright{} (POMDP-inductor): an LLM proposes candidate POMDP models from a few observation-action trajectories and iteratively refines them to optimize a belief-based likelihood score. Despite using strictly less information, \textbackslash emph\textbraceleft Pinductor\textbraceright{} matches the performance and sample efficiency of LLM-based POMDP learning methods that assume privileged access to the hidden state, while significantly surpassing the sample efficiency of tabular POMDP baselines. Further results show that performance scales with LLM capability and degrades gracefully as semantic information about the environment is withheld. Together, these results position language-model priors as a practical tool for sample-efficient world-model learning under partial observability, and a step toward generalist agents in real-world environments. Code is available at https://github.com/atomresearch/pinductor.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Machine Learning},
  file = {/Users/lancelotdacosta/Zotero/storage/EE7LVU6Z/Six et al. - 2026 - Learning POMDP World Models from Observations with Language-Model Priors.pdf;/Users/lancelotdacosta/Zotero/storage/HJG6JTFJ/2605.html}
}


1. pymdp (line 51). It's written as "PyMDP" and "PMDP".
@article{heinsPymdpPythonLibrary2022,
  title = {Pymdp: {{A Python}} Library for Active Inference in Discrete State Spaces},
  shorttitle = {Pymdp},
  author = {Heins, Conor and Millidge, Beren and Demekas, Daphne and Klein, Brennan and Friston, Karl and Couzin, Iain and Tschantz, Alexander},
  year = 2022,
  month = jan,
  journal = {arXiv:2201.03904 [cs, q-bio]},
  eprint = {2201.03904},
  primaryclass = {cs, q-bio},
  urldate = {2022-01-12},
  abstract = {Active inference is an account of cognition and behavior in complex systems which brings together action, perception, and learning under the theoretical mantle of Bayesian inference. Active inference has seen growing applications in academic research, especially in fields that seek to model human or animal behavior. While in recent years, some of the code arising from the active inference literature has been written in open source languages like Python and Julia, to-date, the most popular software for simulating active inference agents is the DEM toolbox of SPM, a MATLAB library originally developed for the statistical analysis and modelling of neuroimaging data. Increasing interest in active inference, manifested both in terms of sheer number as well as diversifying applications across scientific disciplines, has thus created a need for generic, widely-available, and user-friendly code for simulating active inference in open-source scientific computing languages like Python. The Python package we present here, pymdp (see https://github.com/infer-actively/pymdp), represents a significant step in this direction: namely, we provide the first open-source package for simulating active inference with partially-observable Markov Decision Processes or POMDPs. We review the package's structure and explain its advantages like modular design and customizability, while providing in-text code blocks along the way to demonstrate how it can be used to build and run active inference processes with ease. We developed pymdp to increase the accessibility and exposure of the active inference framework to researchers, engineers, and developers with diverse disciplinary backgrounds. In the spirit of open-source software, we also hope that it spurs new innovation, development, and collaboration in the growing active inference community.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Artificial Intelligence,Computer Science - Mathematical Software,Quantitative Biology - Neurons and Cognition},
  file = {/Users/lancelotdacosta/Zotero/storage/YE4NXH2R/Heins et al. - 2022 - pymdp A Python library for active inference in di.pdf;/Users/lancelotdacosta/Zotero/storage/7DBSEIFW/2201.html}
}


2.  Bayes-adaptive POMDP paper (line 51)
3.  

@misc{kattBayesianReinforcementLearning2018,
  title = {Bayesian {{Reinforcement Learning}} in {{Factored POMDPs}}},
  author = {Katt, Sammie and Oliehoek, Frans and Amato, Christopher},
  year = 2018,
  month = nov,
  number = {arXiv:1811.05612},
  eprint = {1811.05612},
  primaryclass = {cs.AI},
  publisher = {arXiv},
  doi = {10.48550/arXiv.1811.05612},
  urldate = {2026-07-29},
  abstract = {Bayesian approaches provide a principled solution to the exploration-exploitation trade-off in Reinforcement Learning. Typical approaches, however, either assume a fully observable environment or scale poorly. This work introduces the Factored Bayes-Adaptive POMDP model, a framework that is able to exploit the underlying structure while learning the dynamics in partially observable systems. We also present a belief tracking method to approximate the joint posterior over state and model variables, and an adaptation of the Monte-Carlo Tree Search solution method, which together are capable of solving the underlying problem near-optimally. Our method is able to learn efficiently given a known factorization or also learn the factorization and the model parameters at the same time. We demonstrate that this approach is able to outperform current methods and tackle problems that were previously infeasible.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Artificial Intelligence},
  file = {/Users/lancelotdacosta/Zotero/storage/VLR8WE56/Katt et al. - 2018 - Bayesian Reinforcement Learning in Factored POMDPs.pdf;/Users/lancelotdacosta/Zotero/storage/QKFQ9IWP/1811.html}
}

@incollection{rossBayesAdaptivePOMDPs2008a,
  title = {Bayes-{{Adaptive POMDPs}}},
  booktitle = {Advances in {{Neural Information Processing Systems}} 20},
  author = {Ross, Stephane and {Chaib-draa}, Brahim and Pineau, Joelle},
  editor = {Platt, J. C. and Koller, D. and Singer, Y. and Roweis, S. T.},
  year = 2008,
  pages = {1225--1232},
  publisher = {Curran Associates, Inc.},
  urldate = {2020-05-05},
  file = {/Users/lancelotdacosta/Zotero/storage/BIQ5P6H2/Ross et al. - 2008 - Bayes-Adaptive POMDPs.pdf;/Users/lancelotdacosta/Zotero/storage/SMSYJQQ5/3333-bayes-adaptive-pomdps.html}
}

@article{rossBayesianApproachLearning,
  title = {A {{Bayesian Approach}} for {{Learning}} and {{Planning}} in {{Partially Observable Markov Decision Processes}}},
  author = {Ross, St{\'e}phane and Pineau, Joelle and {Chaib-draa}, Brahim and Kreitmann, Pierre},
  pages = {42},
  abstract = {Bayesian learning methods have recently been shown to provide an elegant solution to the explorationexploitation trade-off in reinforcement learning. However most investigations of Bayesian reinforcement learning to date focus on the standard Markov Decision Processes (MDPs). The primary focus of this paper is to extend these ideas to the case of partially observable domains, by introducing the Bayes-Adaptive Partially Observable Markov Decision Processes. This new framework can be used to simultaneously (1) learn a model of the POMDP domain through interaction with the environment, (2) track the state of the system under partial observability, and (3) plan (near-)optimal sequences of actions. An important contribution of this paper is to provide theoretical results showing how the model can be finitely approximated while preserving good learning performance. We present approximate algorithms for belief tracking and planning in this model, as well as empirical results that illustrate how the model estimate and agent's return improve as a function of experience.},
  langid = {english},
  file = {/Users/lancelotdacosta/Zotero/storage/6WJA5ME3/Ross et al. - A Bayesian Approach for Learning and Planning in P.pdf;/Users/lancelotdacosta/Zotero/storage/AET8CXI8/Ross et al. - A Bayesian Approach for Learning and Planning in P.pdf;/Users/lancelotdacosta/Zotero/storage/DWW2GVQN/Ross et al. - A Bayesian Approach for Learning and Planning in P.pdf;/Users/lancelotdacosta/Zotero/storage/G4HPPMIQ/Ross et al. - A Bayesian Approach for Learning and Planning in P.pdf}
}

New ref: infinite pomdp 
@inproceedings{oudeyerWhatIntrinsicMotivation2007a,
  title = {The {{Infinite Partially Observable Markov Decision Process}}},
  booktitle = {Advances in {{Neural Information Processing Systems}}},
  author = {{Doshi-velez}, Finale},
  year = 2009,
  volume = {22},
  publisher = {Curran Associates, Inc.},
  urldate = {2025-02-27},
  abstract = {The Partially Observable Markov Decision Process (POMDP) framework has proven useful in planning domains that require balancing actions that increase an agents knowledge and actions that increase an agents reward.  Unfortunately, most POMDPs are complex structures with a large number of parameters.  In many realworld problems, both the structure and the parameters are difficult to specify from domain knowledge alone.  Recent work in Bayesian reinforcement learning has made headway in learning POMDP models; however, this work has largely focused on learning the parameters of the POMDP model.  We define an infinite POMDP (iPOMDP) model that does not require knowledge of the size of the state space; instead, it assumes that the number of visited states will grow as the agent explores its world and explicitly models only visited states.  We demonstrate the iPOMDP utility on several standard problems.},
  file = {/Users/lancelotdacosta/Zotero/storage/4YPWB5SW/Doshi-velez - 2009 - The Infinite Partially Observable Markov Decision Process.pdf}
}


4.  AXIOM (line 51)
5.  
@misc{heinsAXIOMLearningPlay2025,
  title = {{{AXIOM}}: {{Learning}} to {{Play Games}} in {{Minutes}} with {{Expanding Object-Centric Models}}},
  shorttitle = {{{AXIOM}}},
  author = {Heins, Conor and de Maele, Toon Van and Tschantz, Alexander and Linander, Hampus and Markovic, Dimitrije and Salvatori, Tommaso and Pezzato, Corrado and Catal, Ozan and Wei, Ran and Koudahl, Magnus and Perin, Marco and Friston, Karl and Verbelen, Tim and Buckley, Christopher},
  year = 2025,
  month = may,
  number = {arXiv:2505.24784},
  eprint = {2505.24784},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2505.24784},
  urldate = {2025-06-02},
  abstract = {Current deep reinforcement learning (DRL) approaches achieve state-of-the-art performance in various domains, but struggle with data efficiency compared to human learning, which leverages core priors about objects and their interactions. Active inference offers a principled framework for integrating sensory information with prior knowledge to learn a world model and quantify the uncertainty of its own beliefs and predictions. However, active inference models are usually crafted for a single task with bespoke knowledge, so they lack the domain flexibility typical of DRL approaches. To bridge this gap, we propose a novel architecture that integrates a minimal yet expressive set of core priors about object-centric dynamics and interactions to accelerate learning in low-data regimes. The resulting approach, which we call AXIOM, combines the usual data efficiency and interpretability of Bayesian approaches with the across-task generalization usually associated with DRL. AXIOM represents scenes as compositions of objects, whose dynamics are modeled as piecewise linear trajectories that capture sparse object-object interactions. The structure of the generative model is expanded online by growing and learning mixture models from single events and periodically refined through Bayesian model reduction to induce generalization. AXIOM masters various games within only 10,000 interaction steps, with both a small number of parameters compared to DRL, and without the computational expense of gradient-based optimization.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Artificial Intelligence,Computer Science - Machine Learning,Statistics - Machine Learning},
  file = {/Users/lancelotdacosta/Zotero/storage/7KLEQGYV/Heins et al. - 2025 - AXIOM Learning to Play Games in Minutes with Expanding Object-Centric Models.pdf;/Users/lancelotdacosta/Zotero/storage/IXXVKGAY/2505.html}
}

6.  RGM paper, the original (line 53) - for this paper just cite figure 3
7.  @misc{fristonPixelsPlanningScalefree2024,
  title = {From Pixels to Planning: Scale-Free Active Inference},
  shorttitle = {From Pixels to Planning},
  author = {Friston, Karl and Heins, Conor and Verbelen, Tim and Da Costa, Lancelot and Salvatori, Tommaso and Markovic, Dimitrije and Tschantz, Alexander and Koudahl, Magnus and Buckley, Christopher and Parr, Thomas},
  year = 2024,
  month = jul,
  number = {arXiv:2407.20292},
  eprint = {2407.20292},
  primaryclass = {cs, q-bio},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2407.20292},
  urldate = {2024-08-21},
  abstract = {This paper describes a discrete state-space model -- and accompanying methods -- for generative modelling. This model generalises partially observed Markov decision processes to include paths as latent variables, rendering it suitable for active inference and learning in a dynamic setting. Specifically, we consider deep or hierarchical forms using the renormalisation group. The ensuing renormalising generative models (RGM) can be regarded as discrete homologues of deep convolutional neural networks or continuous state-space models in generalised coordinates of motion. By construction, these scale-invariant models can be used to learn compositionality over space and time, furnishing models of paths or orbits; i.e., events of increasing temporal depth and itinerancy. This technical note illustrates the automatic discovery, learning and deployment of RGMs using a series of applications. We start with image classification and then consider the compression and generation of movies and music. Finally, we apply the same variational principles to the learning of Atari-like games.},
  archiveprefix = {arXiv},
  keywords = {92,Computer Science - Machine Learning,F.1.1,Quantitative Biology - Neurons and Cognition},
  file = {/Users/lancelotdacosta/Zotero/storage/63TWHT7K/Friston et al. - 2024 - From pixels to planning scale-free active inferen.pdf;/Users/lancelotdacosta/Zotero/storage/VV3KHJ2E/2407.html}
}

8.  RGM paper by Karim (line 53)
9.  
@misc{zaghwRenormalisingGenerativeModels2026,
  title = {Renormalising {{Generative Models}} for {{Active Inference}}: {{Foundations}}, {{Derivations}}, and {{Verification}}},
  shorttitle = {Renormalising {{Generative Models}} for {{Active Inference}}},
  author = {Zaghw, Karim and Pashea, Andrew and Pritsch, Marc and Nuijten, Wouter and Friston, Karl and Costa, Lancelot Da},
  year = 2026,
  month = aug,
  number = {arXiv:2608.09512},
  eprint = {2608.09512},
  primaryclass = {cs.AI},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2608.09512},
  urldate = {2026-08-18},
  abstract = {Active inference offers a unified framework for perception, learning, and action, but scaling discrete active-inference models to rich spatial and temporal domains remains difficult. Renormalising generative models (RGMs) address this challenge by composing discrete generative models across spatial and temporal scales, coarse-graining lower-level states and paths into higher-level causes for objects, events, and action. However, fully reproducing and adapting the framework remains difficult: the mathematical exposition is compact, and the reference implementations are deeply integrated within specialized software environments, leaving many algorithmic details implicit. This paper addresses these challenges by providing a self-contained, derivation-oriented account of RGMs together with an open, verified implementation. We explain how the hierarchy is built, how beliefs and actions are updated within it, and how information is passed between levels. Where the published equations and implementation differ in emphasis, we make those choices explicit and explain their modelling consequences. By clarifying the theory and separating it from its original implementation context, this work lowers practical barriers to entry and makes RGMs more transparent, auditable, and reproducible, providing a foundation for future quantitative evaluation and development on machine-learning benchmarks.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Artificial Intelligence,Computer Science - Computer Vision and Pattern Recognition},
  file = {/Users/lancelotdacosta/Zotero/storage/CQUVE4SU/Zaghw et al. - 2026 - Renormalising Generative Models for Active Inference Foundations, Derivations, and Verification.pdf;/Users/lancelotdacosta/Zotero/storage/N5GMJCNU/2608.html}
}

10. Thomas Parr paper on hierarchies (line 53)

@article{parrComputationalNeurologyMovement2021,
  title = {The Computational Neurology of Movement under Active Inference},
  author = {Parr, Thomas and Limanowski, Jakub and Rawji, Vishal and Friston, Karl},
  year = 2021,
  month = jun,
  journal = {Brain},
  volume = {144},
  number = {6},
  pages = {1799--1818},
  issn = {0006-8950},
  doi = {10.1093/brain/awab085},
  urldate = {2022-06-17},
  abstract = {We propose a computational neurology of movement based on the convergence of theoretical neurobiology and clinical neurology. A significant development in the former is the idea that we can frame brain function as a process of (active) inference, in which the nervous system makes predictions about its sensory data. These predictions depend upon an implicit predictive (generative) model used by the brain. This means neural dynamics can be framed as generating actions to ensure sensations are consistent with these predictions---and adjusting predictions when they are not. We illustrate the significance of this formulation for clinical neurology by simulating a clinical examination of the motor system using an upper limb coordination task. Specifically, we show how tendon reflexes emerge naturally under the right kind of generative model. Through simulated perturbations, pertaining to prior probabilities of this model's variables, we illustrate the emergence of hyperreflexia and pendular reflexes, reminiscent of neurological lesions in the corticospinal tract and cerebellum. We then turn to the computational lesions causing hypokinesia and deficits of coordination. This in silico lesion-deficit analysis provides an opportunity to revisit classic neurological dichotomies (e.g. pyramidal versus extrapyramidal systems) from the perspective of modern approaches to theoretical neurobiology---and our understanding of the neurocomputational architecture of movement control based on first principles.},
  file = {/Users/lancelotdacosta/Zotero/storage/8RKAUYQN/Parr et al. - 2021 - The computational neurology of movement under acti.pdf;/Users/lancelotdacosta/Zotero/storage/PX2CD2A9/brain-2020-00565-File010.pdf;/Users/lancelotdacosta/Zotero/storage/SUJE6UNJ/Parr et al. - 2021 - The computational neurology of movement under acti.pdf;/Users/lancelotdacosta/Zotero/storage/2IKADJZJ/6168144.html}
}

@article{parrDiscreteContinuousBrain2018a,
  title = {The {{Discrete}} and {{Continuous Brain}}: {{From Decisions}} to {{Movement}}---{{And Back Again}}},
  shorttitle = {The {{Discrete}} and {{Continuous Brain}}},
  author = {Parr, Thomas and Friston, Karl J.},
  year = 2018,
  month = sep,
  journal = {Neural Computation},
  volume = {30},
  number = {9},
  pages = {2319--2347},
  issn = {0899-7667, 1530-888X},
  doi = {10.1162/neco_a_01102},
  urldate = {2019-08-11},
  langid = {english},
  file = {/Users/lancelotdacosta/Zotero/storage/CK25GFGJ/Parr and Friston - 2018 - The Discrete and Continuous Brain From Decisions .pdf;/Users/lancelotdacosta/Zotero/storage/MAMJEMHT/Parr and Friston - 2018 - The Discrete and Continuous Brain From Decisions .pdf}
}

11. ATOM research roadmap blog post: https://blog.lancelotdacosta.com/posts/2026-07-06-atom-research-roadmap/ (line 72)
12. Possible principles for aligned structure learning agents, the "foundational paper" (line 72)
@misc{dacostaPossiblePrinciplesAligned2024,
  title = {Possible Principles for Aligned Structure Learning Agents},
  author = {Da Costa, Lancelot and Gaven{\v c}iak, Tom{\'a}{\v s} and Hyland, David and Samiei, Mandana and {Dragos-Manta}, Cristian and Pattisapu, Candice and Razi, Adeel and Friston, Karl},
  year = 2024,
  month = sep,
  number = {arXiv:2410.00258},
  eprint = {2410.00258},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2410.00258},
  urldate = {2024-11-12},
  abstract = {This paper offers a roadmap for the development of scalable aligned artificial intelligence (AI) from first principle descriptions of natural intelligence. In brief, a possible path toward scalable aligned AI rests upon enabling artificial agents to learn a good model of the world that includes a good model of our preferences. For this, the main objective is creating agents that learn to represent the world and other agents' world models; a problem that falls under structure learning (a.k.a. causal representation learning). We expose the structure learning and alignment problems with this goal in mind, as well as principles to guide us forward, synthesizing various ideas across mathematics, statistics, and cognitive science. 1) We discuss the essential role of core knowledge, information geometry and model reduction in structure learning, and suggest core structural modules to learn a wide range of naturalistic worlds. 2) We outline a way toward aligned agents through structure learning and theory of mind. As an illustrative example, we mathematically sketch Asimov's Laws of Robotics, which prescribe agents to act cautiously to minimize the ill-being of other agents. We supplement this example by proposing refined approaches to alignment. These observations may guide the development of artificial intelligence in helping to scale existing -- or design new -- aligned structure learning systems.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Artificial Intelligence,Quantitative Biology - Neurons and Cognition},
  file = {/Users/lancelotdacosta/Zotero/storage/SBRAQPDK/Costa et al. - 2024 - Possible principles for aligned structure learning.pdf;/Users/lancelotdacosta/Zotero/storage/P5WMF89D/2410.html}
}

