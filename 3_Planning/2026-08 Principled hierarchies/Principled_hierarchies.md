# Principled hierarchies project

## Motivation
Abstraction is currently a fundamental problem in machine learning and world models. We think that learning abstract representations will enable agents to:
- achieve better generalization, since abstract representations generalize to new settings. 
- learn faster, since having good abstractions reduces the number of parameters in the model. 
- Achieve longer-term planning.

It's clear that abstraction should not be imposed a priori, but learned from the data. In an agentic world model setting, that is within the context of ATOM, abstractions are linked to hierarchical models. In a hierarchical world model, there are several layers, where the higher layers are more abstract and the lower layers are closer to the data. We can think of the higher layers as some kind of spatiotemporal coarse-graining of the lower levels.

## Literature scope
Having discussed the importance and the timeliness of learning good abstractions, we know that several fields have approached abstraction learning with different kinds of tools, and it's not clear how the frameworks relate. To name the main ones:
- Hierarchical reinforcement learning, including 
  - various frameworks such as the options framework and goal conditioned RL [chapter 7.4 murphyReinforcementLearningComprehensive2025]
  - recent advances using hierarchical state-space models. The following three references give you three complementary approaches: [zhangHierarchicalPlanningLatent2026,ahmedSynthesizingWorldModels2025a,hafnerDeepHierarchicalPlanning2022]
- Within causal modeling there are hierarchical causal models [weinsteinHierarchicalCausalModels2026]
- Physics and complex systems modeling has a big literature on modeling 
  - multiscale processes [pavliotisMultiscaleMethodsAveraging2010]
  - spatial and temporal coarse-graining, e.g. the renormalization group
- The literature on time series modeling using Bayesian hierarchical state-space models also treats this 
- Related to this, is the literature on hierarchical active inference that has explored several hierarchical types of world models.
  - Latent dynamic models where the latent state at the higher level is an initial condition for the latent trajectory at the lower level [Figure 2, catalRobotNavigationHierarchical2021]
  - Latent dynamic models with a latent state at the higher level relates to the whole trajectory at the lower level [Figure 3, fristonPixelsPlanningScalefree2024]. The most relevant paper in this area is this [zaghwRenormalisingGenerativeModels2026]

## Goal of project
Zooming back, ATOM seeks a space of hierarchical models that is efficiently expressive to express a wide range of worlds, but also sufficiently tractable so that each member of these hierarchical models can carry uncertainty to some extent, engage in inference, learning, and engage in information theoretic planning with information gain. See [costaNaturalBuildingBlocks2025] for a overarching vision paper in this area, and [dacostaPossiblePrinciplesAligned2024] for long version.

The goal of this project is twofold. 
- Theoretical: One, review the aforementioned literatures and assess the synergies and differences. The goal is to understand whether and how these approaches to abstraction learning are connected, deriving theoretical connections.
- Practical: Second, to test the models and assess their learning speed divided by number of samples and compute, to find a space of hierarchical models that actually works well in practice.

## Suggested approach
The suggested approach is to start reading and discuss state-of-the-art papers in hierarchical models of the different approaches, and at the same time implement those models in simple benchmarks to assess the learning speed versus samples and compute. Start with the application. First, think about the project towards the theory. 

## Support team (tentative) 
ATOM will connect you with many experts across the various fields in order to facilitate the research, in particular for
- hierarchical RL experts would be Mohammed Abbas Ansari and Ali Gholamzadeh 
- for hierarchical causal modeling TBD. 
- For multi-scale processes it would be Lancelot Da Costa and, if necessary, Grigorios Pavliotis
- For spatiotemporal coarse graining including renormalisation group it would be Nikos Papanikolaou and Simon Buchholz.
- For hierarchical Bayesian state-space models it would be Conor Heins.
- For hierarchical active inference, the first case, probably Connor Heinz or Wouter Nuijten, and the second case, Karim Zagwh and Andrew Pashea.

## References

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


@misc{fristonPixelsPlanningScalefree2024,
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


@article{catalRobotNavigationHierarchical2021,
  title = {Robot Navigation as Hierarchical Active Inference},
  author = {{\c C}atal, Ozan and Verbelen, Tim and {Van de Maele}, Toon and Dhoedt, Bart and Safron, Adam},
  year = 2021,
  month = oct,
  journal = {Neural Networks},
  volume = {142},
  pages = {192--204},
  issn = {0893-6080},
  doi = {10.1016/j.neunet.2021.05.010},
  urldate = {2021-10-07},
  abstract = {Localization and mapping has been a long standing area of research, both in neuroscience, to understand how mammals navigate their environment, as well as in robotics, to enable autonomous mobile robots. In this paper, we treat navigation as inferring actions that minimize (expected) variational free energy under a hierarchical generative model. We find that familiar concepts like perception, path integration, localization and mapping naturally emerge from this active inference formulation. Moreover, we show that this model is consistent with models of hippocampal functions, and can be implemented in silico on a real-world robot. Our experiments illustrate that a robot equipped with our hierarchical model is able to generate topologically consistent maps, and correct navigation behaviour is inferred when a goal location is provided to the system.},
  langid = {english},
  keywords = {Active inference,Deep learning,RatSLAM,Robot navigation,SLAM},
  file = {/Users/lancelotdacosta/Zotero/storage/T9MIL4MG/S0893608021002021.html}
}


@misc{hafnerDeepHierarchicalPlanning2022,
  title = {Deep {{Hierarchical Planning}} from {{Pixels}}},
  author = {Hafner, Danijar and Lee, Kuang-Huei and Fischer, Ian and Abbeel, Pieter},
  year = 2022,
  month = jun,
  number = {arXiv:2206.04114},
  eprint = {2206.04114},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2206.04114},
  urldate = {2026-02-03},
  abstract = {Intelligent agents need to select long sequences of actions to solve complex tasks. While humans easily break down tasks into subgoals and reach them through millions of muscle commands, current artificial intelligence is limited to tasks with horizons of a few hundred decisions, despite large compute budgets. Research on hierarchical reinforcement learning aims to overcome this limitation but has proven to be challenging, current methods rely on manually specified goal spaces or subtasks, and no general solution exists. We introduce Director, a practical method for learning hierarchical behaviors directly from pixels by planning inside the latent space of a learned world model. The high-level policy maximizes task and exploration rewards by selecting latent goals and the low-level policy learns to achieve the goals. Despite operating in latent space, the decisions are interpretable because the world model can decode goals into images for visualization. Director outperforms exploration methods on tasks with sparse rewards, including 3D maze traversal with a quadruped robot from an egocentric camera and proprioception, without access to the global position or top-down view that was used by prior work. Director also learns successful behaviors across a wide range of environments, including visual control, Atari games, and DMLab levels.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Artificial Intelligence,Computer Science - Machine Learning,Computer Science - Robotics,Statistics - Machine Learning},
  file = {/Users/lancelotdacosta/Zotero/storage/HZH9GGD8/Hafner et al. - 2022 - Deep Hierarchical Planning from Pixels.pdf;/Users/lancelotdacosta/Zotero/storage/ZUA93PNA/2206.html}
}


@misc{zhangHierarchicalPlanningLatent2026,
  title = {Hierarchical {{Planning}} with {{Latent World Models}}},
  author = {Zhang, Wancong and Terver, Basile and Zholus, Artem and Chitnis, Soham and Sutaria, Harsh and Assran, Mido and Balestriero, Randall and Bar, Amir and Bardes, Adrien and LeCun, Yann and Ballas, Nicolas},
  year = 2026,
  month = apr,
  number = {arXiv:2604.03208},
  eprint = {2604.03208},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2604.03208},
  urldate = {2026-04-17},
  abstract = {Model predictive control (MPC) with learned world models has emerged as a promising paradigm for embodied control, particularly for its ability to generalize zero-shot when deployed in new environments. However, learned world models often struggle with long-horizon control due to the accumulation of prediction errors and the exponentially growing search space. In this work, we address these challenges by learning latent world models at multiple temporal scales and performing hierarchical planning across these scales, enabling long-horizon reasoning while substantially reducing inference-time planning complexity. Our approach serves as a modular planning abstraction that applies across diverse latent world-model architectures and domains. We demonstrate that this hierarchical approach enables zero-shot control on real-world non-greedy robotic tasks, achieving a 70\% success rate on pick-\&-place using only a final goal specification, compared to 0\% for a single-level world model. In addition, across physics-based simulated environments including push manipulation and maze navigation, hierarchical planning achieves higher success while requiring up to 4x less planning-time compute.},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Machine Learning},
  file = {/Users/lancelotdacosta/Zotero/storage/LZCKTQ8R/Zhang et al. - 2026 - Hierarchical Planning with Latent World Models.pdf;/Users/lancelotdacosta/Zotero/storage/763T6VCK/2604.html}
}

@misc{murphyReinforcementLearningComprehensive2025,
  title = {Reinforcement {{Learning}}: {{A Comprehensive Overview}}},
  shorttitle = {Reinforcement {{Learning}}},
  author = {Murphy, Kevin},
  year = 2025,
  month = mar,
  number = {arXiv:2412.05265},
  eprint = {2412.05265},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2412.05265},
  urldate = {2025-04-14},
  abstract = {This manuscript gives a big-picture, up-to-date overview of the field of (deep) reinforcement learning and sequential decision making, covering value-based method, policy-gradient methods, model-based methods, and various other topics (e.g., multi-agent RL, RL+LLMs, and RL+inference).},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Artificial Intelligence,Computer Science - Machine Learning},
  file = {/Users/lancelotdacosta/Zotero/storage/JFAEPHD3/Murphy - 2025 - Reinforcement Learning A Comprehensive Overview.pdf;/Users/lancelotdacosta/Zotero/storage/BIKZNBTH/2412.html}
}

@article{weinsteinHierarchicalCausalModels2026,
  title = {Hierarchical {{Causal Models}}},
  author = {Weinstein, Eli N. and Blei, David M.},
  year = 2026,
  journal = {Journal of Machine Learning Research},
  volume = {27},
  number = {37},
  pages = {1--73},
  issn = {1533-7928},
  urldate = {2026-08-18},
  abstract = {Causal questions often arise in settings where data are hierarchical: subunits are nested within units. Consider students in schools, cells in patients, or cities in states. In these settings, unit-level variables (e.g., a school's budget) may affect subunit-level outcomes (e.g., student test scores), and subunit-level characteristics may aggregate to influence unit-level outcomes. In this paper, we show how to analyze hierarchical data for causal inference. We introduce hierarchical causal models, which extend structural causal models and graphical models by incorporating inner plates to represent nested data structures. We develop a graphical identification technique for these models that generalizes do-calculus. We show that hierarchical data can enable causal identification even when it would be impossible with non-hierarchical data--for example, when only unit-level summaries are available. We develop estimation strategies, including using hierarchical Bayesian models. We illustrate our results in simulation and through a reanalysis of the classic "eight schools" study.},
  file = {/Users/lancelotdacosta/Zotero/storage/HXINX2VR/25-0899.html}
}

@book{pavliotisMultiscaleMethodsAveraging2010,
  title = {{Multiscale Methods: Averaging and Homogenization}},
  shorttitle = {{Multiscale Methods}},
  author = {Pavliotis, Grigoris and Stuart, Andrew},
  year = 2010,
  month = nov,
  edition = {Softcover reprint of hardcover 1st ed. 2008 \'edition},
  publisher = {Springer},
  address = {New York, NY},
  abstract = {This introduction to multiscale methods explores both theory and applications. Examples show how to apply multiscale methods to solve a variety of problems. Exercises then enable readers to build their own skills and put them into practice.},
  isbn = {978-1-4419-2532-9},
  langid = {Anglais}
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
