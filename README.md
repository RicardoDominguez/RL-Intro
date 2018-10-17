# Reinforcement Learning

Repository dedicated to the implementation of several Reinforcement Learning methods in Python, exemplified by OpenAI environments. Some performance comparisons are included.

Most implementations are based on David Silver's UCL RL course and Sutton and Barto's book Reinforcement Learning: an Introduction.

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

Several classes and functions required for the above files are contained in util.py. 

The environment file gridworld.py might be required for some examples and comparisons.

Sources used:
 * David Silver (2015), COMPM050/COMPGI13 Lectures.
 * Sutton and Barto (2012), Reinforcement Learning: an Introduction.
 * Csaba Szepesvari (2009), Algorithms for Reinforcement Learning.
 * Daniel Takeshi (2016), Going Deeper Into Reinforcement Learning: Understanding Q-Learning and Linear Function Approximation.
 * David Silver, Richard Sutton and Martin Muller (2012).
