# Research agenda: uncertainties within the scientific field

## Graphical models versus probabilistic programs versus LLMs

## Optimization scheme

- In relation to potential universality of free energy, if all schemes minimise FE it begs the question whether we want to minimise FE explicitly, or implicitly through another set of objectives.
    -> Bayesian approaches or other approaches [current paper will shed light]
- That being said, uncertainty is crucial for active learning via intrinsic motivation.
    - But can uncertainty itself be amortised? cf LLM agents - bypassing the need for explicit Bayesian approaches altogether.

## Priors: Hard-coded or learned

- Hard-coded: Core knowledge priors.
- Learned: LLMs for Program Synthesis, RL agent training with lots of interactions (offline or online), evolutionary search for program synthesis.
- Soft core priors: AG Wilson perspective [where does it fit in this?]

## Space of models

- Explicit and sparse models or dense transformer models (in-context learning)
    - Explicit models: separation of temporal scales, goals and subgoals, multiple levels of abstraction.
    - Explicit models: tree structure or network structure? [todo]
    - Dense models: attention heads + network structure -> compositional generation & in-context Learning (=> higher compression).
    - Dense models: highly non-Markovian - long range history contribution
- What is the right space of models?
    - Candidate substrates
        - Causal models
        - Probabilistic graphical models
            - PGMs = causal models with intervention nodes [causality with gates]
        - Programs e.g. probabilistic
            - Programs = PGMs, but question is what subset to consider:
                - Balancing expressiveness VS fast inference over model class: how far and broad can we push fast inference techniques?
        - Neural networks
            - Are these models or estimators for implicitly defined models?
            - Can potentially implement a much richer class of (VI) algorithms and representations? [c.f ryan's work]
            -> Black-box inference
        -> PGMs/programs/causal models VS neural nets.
    - Expressiveness
        - Theory of computation: scoring expressiveness - what distributions can we compute?
    - Equivariance -> further compression.
    - Compositional generation:
        - Compositional generalisation & in-context learning
        - favors group structure within the model (e.g. vector space representations/continuous representations?).

## Assessing performance: the Measure of Intelligence

- If free energy is unable to assess intelligence for structure learning agents, then what scores?
- Several candidates to investigate:
    - Chollet's measure of intelligence: his paper & ARC Challenge & related thoughts,
    - Compression and intelligence
    - Reward in broader set of tasks. Which?
        - Task is unseen to developer (prevents task overfitting)
- Other important metrics
    - Sample efficiency
    - Compute/energy efficiency (intelligence per kWh)
    - Transfer learning and OOD/compositional generalisation/extrapolation

## Causal representation learning vs deep learning

CRL <=> sparsity. What good is sparsity?
- lower (computational) complexity, broader generalisation?
- better compression?
(small worldness of networks)
