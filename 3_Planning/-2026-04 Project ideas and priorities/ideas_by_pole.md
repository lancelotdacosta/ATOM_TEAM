# ATOM Research Projects by Pole

## 1. World Modeling Pole

### 1.1 Theory

#### Model Space Stack: Programs vs Graphical Models vs Causal models vs Neural nets
- Three Competing Stacks for Uncertainty & Architecture: *Q: What is the right space of models? (Causal models vs PGMs vs Programs vs Neural networks?)* -> connections and differences between these - need a review?
- *Universal modelling: developing increasingly expressive hypothesis space*
- *Desiderata tension: How to balance expressiveness, tractability, and fast action/perception/learning in model spaces?*
- *World models as probabilistic programs: How to achieve universally expressive yet computationally tractable model spaces?*
- *World model learning: programs or graphical models as computational substrate?*
- Challenge: finding expressive enough sets of probabilistic graphical models as hypothesis space for world models
- *Programs advantages: expressive hypothesis space, ability to express algorithmic/compositional/recursive knowledge, compositional generalization*
- Programs disadvantages: inference requires sampling (slow)
- Graphical models advantages: fast variational inference (message passing), local and biologically plausible update rules
- PGMs and probabilistic programs equivalence exploration
- Causal models versus programs in humans

#### Deep Learning and LLMs
- What is the difference between learning unstructured generative models and classic deep learning? Is it explicit complexity minimization?
- *LLM interpretability vs explicit program synthesis/model inference*
- Q: Is the transition distribution an attention head?
- Attention as implicit structural inference
- Autoregressive modelling approaches
- LLMs & Transformers architecture optimization
- Mamba and Selective SSMs
- Representation learning in deep networks
  
#### Model space expressiveness and theory of computation
- Turing complete hypothesis space and prior for simplicity:
  - If we want to be able to model/understand fairly arbitrary situations, and we want a system that's as general as the brain, then observe that the brain can follow through arbitrary computer programs: the brain can express a Turing complete hypothesis space (barring memory limitations). Therefore we need a system that entertains a Turing complete hypothesis space (barring such limitations).
  - Recent theoretical results suggests this underpins the success of transformers: a universally expressive hypothesis space and a prior for simplicity (ie compression) prevents overfitting (ie benign overfitting).
  - Embracing universal hypothesis space means embracing large computational requirements.
  - Why Turing Completeness? -> Brain can follow arbitrary computer programs -> Turing complete hypothesis space -> Need system with comparable expressiveness for general intelligence -> Enables modeling of fairly arbitrary situation
- *Computational power of Dynamic Bayesian Networks - what classes of Bayesian networks are Turing complete*
- Brain as Turing complete system (barring memory limitations)

#### Renormalizable Model Spaces and Scale-free-ness
- To what extent is the world renormalisable? What is the renormalization group? What is its scope? When does it apply?
- How does a scale-free architecture account for a non-scale-free world? Connection to RGMs
- Should RGMs be renormalizable in time (e.g., for language)?
- *How universal are renormalising generative models?*
  - RG flow: https://www.damtp.cam.ac.uk/user/tong/sft/three.pdf
  - Connection to scale space models and rg papers from the literature
- Q: Is the best model truly sparse? (Physical dynamics sparsity vs scale-free world view)
  - Beren's insight: The world is somehow scale-free and the best model will not necessarily be sparse - a small number of variables can explain large variance in data, but explaining small-order effects requires many extra connections (possibly with small weights)
- Small worldness of networks

#### Compositional and OOD Generalization - link with compression
- *Does a generative model trained with complexity regularizer automatically learn latent features for compositional generalization without explicit compositional search at test time?*
- Compositional search from building blocks vs learning features that generalize through compression
- Both structured and unstructured models can achieve compositional generalization if compression is good?
- Compositional structure emerges from compression?
- Structure of language and compositional generation
- Transfer learning and OOD/compositional generalisation/extrapolation

### 1.2 Methods

#### Hierarchical modeling
- Hierarchical POMDP
- Hierarchical/renormalizable axiom
- Review of hierarchical RL literature/frameworks

#### Bayesian Parameter Learning
- Bayesian learning of HMM/POMDP parameters
- Limitations of Dirichlet-count update
- Joint AB learning in RGM possible? Is it better suited than POMDPs?
- Generative model discovery factorization: structure learning → functional form/parameter learning → parameter fitting → state inference

#### Continuous primitives 
- Polishing/improving upon rslds-based pomdp

#### Automated Inference Algorithm Discovery
- Automated message passing/VI discovery
  - e.g. with LLMs AlphaEvolve style

### 1.3 Engineering

(To be populated as needed)

## 2. Model Discovery Pole

### 2.1 Theory

#### Information Geometry of Models
- Embedding of models with Information Geometry
- *Geometry of spaces of models/programs & dynamics (search/inference/optimisation)*
- Geometry of POMDPs/Bayesnets/trees/DAGs/architectures
- Embedding of DAGs/processes in continuous space
- Neural Spacetimes for DAG Representation Learning
- Geometry of models exploration
- Ginzburg Landau models: most generic kind of non-linear system? Can it be used to navigate complex FE landscapes?

#### Model Comparison and Learning Theory
- Structured vs unstructured models as a Bayesian model comparison problem
- Rich priors vs no priors: How to disentangle this from the structured vs unstructured debate?
- *From Overfitting to Generalization: The Need for Good Priors in Model Discovery*
- *In-context learning vs structure learning*: What is the relationship to model discovery?
- Attention as implicit structural inference
- PAC-Bayesian Generalisation Bounds
- No free lunch theorems for structure learning

#### Priors for Sample-Efficient Learning
- *Compression, complexity minimization, and intelligence: Does F ≥ -log p(d) imply that compression/MDL leads to OOD/compositional generalization?
- Is compression sufficient for general intelligence? What is the relationship?
- Compression-based priors + prior for simplicity
- Do structured models inherently compress better than unstructured ones, or is compression ability independent of structure?
- Sparsity benefits: lower complexity, broader generalisation, better compression
- *Core knowledge priors (Spelke): reverse engineering those priors*
- Core priors: renormalisation, object centric, equivariance
- POMDP vs object centric for human representations, cf core knowledge
- Soft domain-specific priors for efficiency gains when domain knowledge available
- Bitter lesson: generality -> use general priors and harness large computation

### 2.2 Methods

#### Structure Learning (Passive)
- Generative model discovery factorization: structure learning → functional form/parameter learning → parameter fitting → state inference
- *efficient VI over models*
- Non-identifiability problem: Multiple models explain same observational data equally well
- Joint (observations + actions) data requirement for finding good world models
- Causal models for scientific discovery

#### Active Markov Blanket detection
- Hierarchical nested Markov blankets as universal data structure for world modeling
- Markov blanket discovery in continuous state-space (SLA + FSL approaches)
- Dynamic Markov Blanket Detection (Beck & Ramstead 2025)
- *Doing model discovery before param or state inference -> using causality to actively determine structure before fitting parameters*
- Infer factorisation (graphical) structure through conditional independence tests (latent edges)

#### GFlowNets
- GFlow Nets as amortized variational inference: How do they explore multimodal posteriors?
- GFlow nets training as variational inference - can this be made real-time?
- Trade-off: extremely accurate posteriors vs computational speed
- Real-time capability of GFlow nets
- Gflownets and hierarchical pomdps

#### Program Induction/Bayesian Model Discovery/Structure Learning Algorithms
- *Probabilistic model inference (Bayesian model discovery)*
- *Model discovery: inferring plausible models online with suitable priors*
- *Universal Generative Models (UGMs) and Renormalizing Generative Models (RGMs) as foundation for structure learning*?
- *Efficient structure learning algorithms that enable real-time decision-making (vs expensive traditional approaches)*
- *Differentiable structure learning, non-parametric Bayes, Bayesian optimization for structure discovery*
- *Fast structure learning as approximation to hierarchical Dirichlet process*
  - Benchmarking against hierarchical Dirichlet process for not necessarily adding new states when encountering new observations
- Differentiable architectural search? for structure learning?
- Testing structure learning on increasingly complex environments (TMaze, multiple degrees of freedom for structure)
- LLM guided POMDP search/inference // Alphaevolve
  - From word models to world models
- Review of program search/program synthesis/symbolic gradient descent etc

### 2.3 Engineering

(To be populated as needed)

## 3. Decision-Making Pole

### 3.1 Theory

#### Uncertainty and decision-making
- Q: Should we minimise free energy explicitly or implicitly through another set of objectives?
- *Q: Can uncertainty itself be amortised? (cf LLM agents - bypassing explicit Bayesian approaches)*
- Q: When is maintaining uncertainty useful? (Beyond exploration - what else?)
- What things do we have uncertainty over, or what things shouldn't we have: cf bounded rationality
- Iterative inference vs amortized inference distinction
- Epinet- Epistemic Neural Networks for uncertainty quantification -is there a way of using this for world models with uncertainty, underpinning info gain? For vast efficiency gains compared to using ensembles
- Diffusion models with uncertainty quantification

### 3.2 Methods

#### Reinforcement Learning
- Offline RL for unlocking good priors
- SOTA reinforcement learning approaches
- Model-based reinforcement learning integration
- Safety: What Richard Sutton said
- *Bayes-Adaptive POMDPs vs Expected-Free-Energy Planning comparison*
- Sophisticated Inference vs BA-POMDPs bridging
- Deep Active Inference Agents for Delayed and Long-Horizon Environments

#### Sophisticated inference with tree search
- Sophisticated inference with tree search: showing it seeks information relevant only for reward

#### Active inference and artificial reasoning
- Active inference and artificial reasoning: Translating, understanding and scaling

#### Active Structure Learning / Experimental Design
- *Active structure learning: Bayesian inference over structure/parameters/states with info gain-driven action selection, decade review of approaches*
- *Active Bayesian Causal Inference*
- Why didn't early active structure learning approaches (Infinite POMDP, Active Causal Learning) take off? Can modern advances overcome those limitations?
- Infinite POMDP approaches (Doshi-Velez 2009) - revisiting why they didn't take off and how to overcome limitations
- Infinite POMDP: improving with modern machinery
- *Active learning: selecting actions to gain information about the world*
- *Information-theoretic action selection*
- *Hypothesis testing through active environment interaction*
- Experimental design routine that proposes experiments to disambiguate candidates

### 3.3 Engineering

(To be populated as needed)

## 4. Engineering Pole

#### Agent Architectures and Systems
- Agent foundation model: An agent that efficiently learns in any environment, in an unbounded way
- Model synthesizer + agent architecture: Generate diverse candidates + select informative actions

#### Scale and Hardware
- *Scale and parallel GPU training*
- *GPU parallelization for large-scale structure discovery*
- Hardware-aware algorithms for structure learning (cf. Mamba)
- *Sparse matrices on modern GPUs: Performance benefits vs dense matrices?*

#### Systems-Level Trade-offs
- *Trade-off between strong priors (efficient but bottleneck) vs weak universal priors with lots of data*

#### Benchmarks and Evaluation
- *Showing supremacy on something*
- *Where structure learning provides advantages: sample efficiency, compute efficiency, OOD generalization*
- *Testing on unseen tasks unknown to developers as the "holy grail" of ML*
- ARC Challenge variants, Animal-AI, MuJoCo, DeepMind Lab as potential benchmarks
- *Post-ARC-AGI: Agentic benchmarks for flexible learning on unseen tasks*
- Reward in broader set of tasks (unseen to developer)
- Task overfitting prevention
- *Sample efficiency metrics*
- Compute/energy efficiency (intelligence per kWh)

## 5. Exploration Pole

#### Thousand Brains Theory
- 1000 brains theory: 150k cortical columns with same data structure and learning algorithm
- Thousand brains verification: Do cortical columns represent different world model hypotheses?
- Q: How do these models co-evolve? (Variational inference?)
- Q: Do particles evolve independently or together? (Stein variational gradient descent vs GFlow Net gradient descent)
- Connection to thousand-brain theory: if cortical columns evolve via Stein variational gradient descent or GFlow Net gradient descent or something else
- Q: How are cortical columns connected and how do they communicate?
- Neo-cortex: 150k cortical columns with unique architecture for vision, language, touch → huge inductive bias
- Parallel processing of hundreds of thousands of hypotheses at any time
- Working memory maintains small number of hypotheses while brain processes hundreds of thousands in parallel
- Connection to particle-based inference: small particles in consciousness vs massive parallel unconscious processing
- Unified neocortical algorithm for multimodal structure learning (vision, language, touch)
- Preprocessing and voting/aggregation mechanisms in multi-modal architectures (cf. cortical columns)
- Attention mechanism over unconscious hypothesis space
- Multiple models in consciousness (small number) as attention operator on thousands of unconscious models
- Implementation of thousand brains theory of intelligence

#### Causal Representation Learning and Reinforcement Learning
- CRL & RL: Theoretically, the relationship - are they the same?
  - cf Pearl's causal framework and PGMs with action nodes equivalence investigation
- Deep learning and causal representation learning: relationship and integration
- Continuous relaxation of connection weights in deep learning
- Q: How is optimization done in causal representation learning?

#### Intelligence Measurement
- Intelligence measurement: Chollet's framework, role of action, post-ARC-AGI benchmarks

#### Language and RGMs
- RGM for language: Is prompting equivalent to specifying the first few observations?
- Improving on: synthetising world models for bilevel planning - LLM/program synthesis front

#### Hierarchical RL
- Improving Danijar's director: Deep Hierarchical Planning from Pixels

#### Cognitive Development
- Causal learning vs probabilistic program learning as models of cognitive development
- Building models of the world through cognitive development
- Child as hacker: minimizing model complexity
- Child as scientist: hypothesis testing
- Child as active causal learner: interventions and experiments
- How these perspectives inform structure learning algorithms
- Active structure learning in children (Gopnik): empirical data and insights

#### Neural Implementation of Structure Learning
- Brain development and structure learning: How does developmental pruning/growing inform model expansion? Neurobiological basis (Isomura et al.)
- Structure learning in neuroscience:
  - Bridge between neural dynamics and VFE minim: can we have a similar bridge for structure learning?
  - Insights on how neurons perform structure learning: the update algorithms and the type of structures they encode. Empirical data from Cortical Labs
  - How the brain changes during development: structure; evidence for growing or pruning?
- Adding neurons in connection between neural dynamics and variational inference: what happens then to the associated POMDP? Incorporating models of morphogenesis.
- Bridge between neural dynamics and active inference for structure learning
- Q: Are neural networks models or estimators for implicitly defined models?

#### AI for Science
- Drug discovery applications
- Materials science applications
- Scientific model discovery

#### Commercial Applications
- Genius-level problem solving and on-the-fly model discovery
- Technology that can be deployed anywhere and self-improves by gathering its own data
- Not amortized behavior across pre-defined situations, but true novel input handling
- On the fly model creation for wearable devices

#### Following up on Pinductor
- Use pinductor + prompt 'likely partially observed so observation model likely identity'
- LLM guided planning for solving different problems based on Mathis' branch
- Qwen 3.6 27B fits competition requirements - 1gpu 12hs
- Look at frontier performance which already uses program synthesis:
  - https://github.com/astroseger/arc-3-agents-baseline1
  - https://schema-harness.github.io