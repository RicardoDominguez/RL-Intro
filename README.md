# Reinforcement Learning

Repository dedicated to the implementation of several Reinforcement Learning methods in Python, exemplified by OpenAI environments. Some performance comparisions are included.

Currently implemented:
* agentTabular.py: Tabular action value methods.
  * SARSA
  * SARSA(λ)
  * Q-learning
  * Watkins Q-learning

* agentIncrementalVFA.py: Incremental methods using value function approximation.
  * TD
  * TD(λ)
  * Gradient TD2
  * Gradient Q-learning
  * Recursive Least Squares TD
  
* agentBatchVFA.py: Batch methods using value function approximation.
  * Least Squares TD
  * Least Squares TD(λ)
  * Least Squares TDQ
  * Least Squares Policy Iteration TD
  
* agentActorCritic.py: Actor-critic methods.
  * Q Actor-Critic
  * Advantage Actor-Critic
  * TD Actor-Critic
  * TD(λ) Actor-Critic

* agentModelBased.py: Model based methods.
  * DynaQ
  * Monte Carlo Tree Search
  * TD Tree Search
  * Dyna2
