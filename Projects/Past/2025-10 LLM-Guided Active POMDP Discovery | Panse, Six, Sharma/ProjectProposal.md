# Project Proposal

## **LLM-Guided Active Model Discovery with Parallel Hypothesis Testing**

### Executive Summary
We are proposing an ambitious collaborative research project that combines cutting-edge approaches from machine learning, neuroscience, and Bayesian inference. This project aims to improve how AI systems autonomously discover, test and refine models of their environment, with potential applications ranging from automated scientific discovery, cognitive modelling and autonomous AI agents.

### Core Research Question
**How can we build AI agents that autonomously discover accurate models of complex environments?**

### Approach
Build an agent that actively learns a model of the world by combining:
- LLM-generated hypothesis (i.e. model) generation
- Brain-inspired parallel evaluation
- Information-seeking actions via experimental design

### Technical Architecture

#### **System Architecture**

This is an active learning loop that consists of the following two steps performed repeatedly in a sequence:

• **Hypothesis Generation and refinement Module**
  - Leverage LLMs to propose *diverse* POMDP candidate models (probabilistic graphical models)
  - Inspired by the brain's 150,000 cortical columns that simultaneously maintain different world models
  - Test these models in parallel against the data
  - Potentially refine each hypothesis with the LLM

• **Active Learning Controller** 
  - Each hypothesis maintains its own belief state and makes predictions
  - Bayesian framework for experiment/action selection
  - Calculate expected information gain for potential actions
  - Select experiments that maximally distinguish between competing hypotheses

#### **Key Features**

• **Parallel vs Sequential Testing**: Unlike traditional approaches that test hypotheses one-by-one, this system evaluates several simultaneously
• **Neuroscience-Inspired Design**: Design is inspired by how cortical columns in the brain maintain multiple models
• **Core priors and active learning**: The system leverages learned knowledge inside LLMs and information-seeking actions to maximize sample efficiency
• **LLM-Guided Search**: Uses language models to navigate the vast space of possible models more efficiently than random or exhaustive search
• **Deep learning guided program synthesis**: Bridges symbolic (programs/graphs) and neural (LLM) approaches to model discovery

#### **Key development components**

1. **LLM for generating POMDPs**: Using language models to propose *diverse* POMDP candidate models
2. **POMDP inference and control**: Implementing inference algorithms and control strategies for the generated models
3. **Active learning component for decision-making**: Information-theoretic action selection to maximize learning efficiency
4. **Connection to Thousand Brains Theory** (optional and cosmetic): Parallel hypothesis testing inspired by cortical columns

### Development Approach

**Starting Point**: The first step is to thoroughly read and understand the "LLM-Guided Probabilistic Program Induction for POMDP Model Estimation" paper (linked in the relevant references). Understanding this work deeply and identifying potential improvements is crucial - *any meaningful improvement to this paper would constitute a publication in itself and could be the minimal output for this project.*

**Development Strategy**: 
• Begin by developing a proof of concept and working prototype based on insights from the foundational paper
• Iterate continuously to improve the prototype over time
• Focus on incremental enhancements that can each contribute to the final system

**Benchmarks**: 
• Start by examining the benchmarks used in the LLM-POMDP paper
• Explore additional benchmarks suitable for our extended approach
• Note that identifying the right benchmark for this work constitutes an important part of the research project itself

### Expected Outcomes

• **Primary Deliverable**: Publication demonstrating the approach (aiming for ICML/NeurIPS but can pivot to lower-tier venues depending on how the project goes)
• **Software Release**: Open-source codebase with documentation
• **Follow-up Opportunities**: Foundation for future work in AI agents and/or scientific discovery, reinforcement learning, LLMs, program synthesis
• **Long-term Impact**: This ambitious project may extend beyond individual thesis timelines. Contributors who make substantial contributions will be included as co-authors even if the project continues after their departure

### Why This Project Matters

• **Scientific Impact**: Advances understanding of how to efficiently discover models in complex domains
• **Practical Applications**: Direct applications to scientific discovery and autonomous systems
• **AGI Relevance**: Core capability needed for generally intelligent systems that can adapt to new environments

### Career development

• Positions contributors at the intersection of multiple hot research areas (LLMs, neuroscience-inspired AI, program synthesis, active learning).
• The experience itself will be valuable on your CV, as research experience in one of the top machine learning labs in the world.
• Contributors can count on a recommendation letter from me (and I will do what I can to get one from Bernhard Schölkopf as well) for grad school/other applications following the project.

### Collaboration and Learning Structure

• **Regular Group Meetings**: Regular coordination and deep technical dives. Technical dives will alternate between this project and on other projects led by other students/postdocs relevant to the Agent Foundation Model research program [https://www.lancelotdacosta.com/research](https://www.lancelotdacosta.com/research).
• **Expert Connections**: I will connect contributors with other experts in the field as needed for project progress, either by inviting them to our group meetings or facilitating separate introductions
• **Peer Collaboration**: Contributors are expected to work collaboratively outside of our meetings. The aim is not to hinder the possibility of individual work, and if someone works much better individually most of the time, this should be prioritized. What is important is for us to find the organization that best fastens the pace of progress, and in particular not reinvent someone else's wheel. The team has been assembled with complementary skills and social compatibility in mind - and as a general rule of thumb, more collaboration while maintaining long, non-interrupted stretches of deep work increases the chances of success.
• **Leadership & Mentorship**: Since this project has multiple components, individuals will have opportunities to lead areas that suit their strengths. You'll be expected to mentor others on your areas of expertise while being mentored on components where others are stronger.
• **Shared Repository**: Collaborative development
• **Shared Tools**: Notes in markdown, manuscript and maths in latex and core development in python
• **Joint Authorship**: All major contributors as co-authors on publications - substantial contributions will be recognized even if you leave before project completion
• **Skill Exchange**: Opportunities to learn from peers' complementary expertise

### Relevant References

#### **LLM Component**
• LLM-Guided Probabilistic Program Induction for POMDP Model Estimation: [https://arxiv.org/abs/2505.02216](https://arxiv.org/abs/2505.02216)
• AlphaEvolve: A coding agent for scientific and algorithmic discovery: [https://arxiv.org/abs/2506.13131](https://arxiv.org/abs/2506.13131)

#### **Neuroscience-Inspired and Active Learning Component**
• Possible Principles for Aligned Structure Learning Agents (Sections 3 and 4): [https://arxiv.org/abs/2410.00258](https://arxiv.org/abs/2410.00258)

#### **Active Learning Foundations**
• Active Learning of Causal Structure: [https://www.cs.ubc.ca/~murphyk/Papers/alearn.pdf](https://www.cs.ubc.ca/~murphyk/Papers/alearn.pdf)
• The Infinite Partially Observable Markov Decision Process: [https://papers.nips.cc/paper_files/paper/2009/file/ebd9629fc3ae5e9f6611e2ee05a31cef-Paper.pdf](https://papers.nips.cc/paper_files/paper/2009/file/ebd9629fc3ae5e9f6611e2ee05a31cef-Paper.pdf)

#### **POMDP Inference and Control**
• Planning and acting in partially observable stochastic domains: [https://pdf.sciencedirectassets.com/271585/1-s2.0-S0004370200X00410/1-s2.0-S000437029800023X/main.pdf](https://pdf.sciencedirectassets.com/271585/1-s2.0-S0004370200X00410/1-s2.0-S000437029800023X/main.pdf)
• Bayes-Adaptive POMDPs: [https://proceedings.neurips.cc/paper/2007/hash/3b3dbaf68507998acd6a5a5254ab2d76-Abstract.html](https://proceedings.neurips.cc/paper/2007/hash/3b3dbaf68507998acd6a5a5254ab2d76-Abstract.html)
• Active inference on discrete state-spaces: A synthesis: [https://www.sciencedirect.com/science/article/pii/S0022249620300857](https://www.sciencedirect.com/science/article/pii/S0022249620300857)

### Important Notes

• **Team Requirement**: **This project is contingent upon at least two students accepting to work on it**.  Given the multiple The collaborative nature seems essential to its success
• **Project Evolution**: While ambitious in scope, the project is not set in stone and may evolve based on progress, discoveries, and team insights
• **Growing Team**: New contributors may join the project as it progresses, creating opportunities for expanded collaboration, learning and faster progress

### Next Steps

Review this proposal and let me know if you're interested in working on this project:

*Optionally:* list any questions/discussion points for the next meeting, which may include: which aspects of the project are most exciting to you, are there specific components you'd be most interested in contributing to, what additional ideas or modifications would you suggest, what concerns or challenges do you foresee. Think about potential first steps, extensions or variations. Consider which components align best with your expertise.

This project offers a unique opportunity to work on cutting-edge research that bridges multiple disciplines while making concrete progress toward more intelligent and efficient AI systems. We believe the combination of different perspectives and skills will lead to results beyond what one of us could achieve individually.