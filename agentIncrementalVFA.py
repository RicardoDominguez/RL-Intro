# Implementation of the following incremental methods for value function approximation
# using linear combination of features and table lookup features:
#   - TD(0) [1]
#   - TD(lamda) [1]
#   - Gradient TD2 [2]
#   - Gradient Q-learning [3]
#   - Recursive Least Squares TD [4]
# to be used with OpenAI Gym environments. Demonstrations are included with the
# following environments: GridWorld-v0.
#
# [1] - David Silver (2015), COMPM050/COMPGI13 Lecture 6, slide 22
# [2] - Csaba Szepesvari (2009), Algorithms for Reinforcement Learning, page 35
# [3] - Daniel Takeshi (2016), Going Deeper Into Reinforcement Learning:
#   Understanding Q-Learning and Linear Function Approximation
# [4] - Csaba Szepesvari (2009), Algorithms for Reinforcement Learning, page 37
#
# By Ricardo Dominguez Olmedo, Aug-2017

# Import necessary libraries and functions
import numpy as np
from util import Agent
from util import Featurize
from util import LinearVFA
from util import EGreedyPolicyVFA

# Implements the specific functionality of a incremental value function approximation
# agent, such as to initialize the agent or run episodes.
class AgentIncrementalVFA(Agent):
    def __init__(self, env, policy, VFA, featurize, alpha, beta = 0.2,
        lamda = 0, gamma = 1, horizon = 1000, verbosity = 0):
        # Inputs:
        #   -env: openAI gym environment object
        #   -policy: object containing a policy from which to sample actions
        #   -VFA: object containing the value function approximator
        #   -featurize: object which featurizes states
        #   -alpha: step size parameter
        #   -beta: secondary step parameter used in GTD2 and RLSTD
        #   -lamda: trace discount paramater
        #   -gamma: discount-rate parameter
        #   -horizon: finite horizon steps
        #   -verbosity: if TRUE, prints to screen additional information

        self.env = env
        self.policy = policy
        self.featurize = featurize
        self.VFA = VFA
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma
        self.beta = beta
        self.horizon = horizon
        self.verbosity = verbosity

        self.nS = env.observation_space.n   # Number of states
        self.nA = env.action_space.n    # Number of actions
        self.policy.setNActions(self.nA)
        self.featurize.set_nSnA(self.nS, self.nA)
        self.featDim = featurize.featureStateAction(0,0).shape # Dimensions of the
                                                               # feature vector
        self.VFA.setUpWeights(self.featDim) # Initialize weights for the VFA

        # Initialize other weights used by GTD2 and RLSTD
        self.auxWeights = np.ones(self.featDim) # used by GTD2
        self.weightsRLSTD = np.eye(self.featDim[0]) * beta # used by RLSTD

        # Initially prevent agent from learning
        self.learn = 0

    # Computes a single episode.
    # Returns the episode reward return.
    def episode(self):
        episodeReward = 0
        self.setUpTrace()

        # Initialize S, A
        state = self.env.reset()
        action = self.policy.getAction(self.VFA, self.featurize, state)

        # Repeat for each episode
        for t in range(self.horizon):
            # Take action A, observe R, S'
            state, action, reward, done = self.step(state, action)

            # Update the total episode return
            episodeReward += reward

            # Finish the loop if S' is a terminal state
            if done: break
        # Update the policy parameters if the agent is learning
        if self.learn: self.policy.episodeUpdate()

        return episodeReward

    # Initilize trace matrix
    def setUpTrace(self):
        self.E = np.zeros(self.featDim)

# Implementation of the TD(0) incremental method for value function approximation
class TD(AgentIncrementalVFA):
    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        action_prime = self.policy.getAction(self.VFA, self.featurize, state_prime)

        # If the agent is learning, update the VFA weights using TD(0)
        if self.learn:
            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)

            # Compute the value of the features via value function approximation
            value = self.VFA.getValue(features)
            value_prime = self.VFA.getValue(features_prime)

            # Update the VFA weights
            delta_w = (self.alpha * (reward + self.gamma * value_prime - value)
                * self.VFA.getGradient(features))
            self.VFA.updateWeightsDelta(delta_w)

        return state_prime, action_prime, reward, done

# Implementation of the TD(lamda) incremental method for value function approximation
class TDlamda(AgentIncrementalVFA):
    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        action_prime = self.policy.getAction(self.VFA, self.featurize, state_prime)

        # If the agent is learning, update the VFA weights using TD(lamda)
        if self.learn:
            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)

            # Compute the value of the features via value function approximation
            value = self.VFA.getValue(features)
            value_prime = self.VFA.getValue(features_prime)

            # Update the VFA weights
            delta = reward + self.gamma * value_prime - value
            self.E = (self.gamma * self.lamda * self.E) + self.VFA.getGradient(features)
            delta_w = self.alpha * delta * self.E
            self.VFA.updateWeightsDelta(delta_w)

        return state_prime, action_prime, reward, done

# Implementation of the Gradient TD2 incremental method for value function approximation
class GradientTD2(AgentIncrementalVFA):
    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived S'
        action_prime = self.policy.getAction(self.VFA, self.featurize, state_prime)

        # If the agent is learning, update the VFA weights using GTD2
        if self.learn:
            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)

            # Compute the value of the features via value function approximation
            value = self.VFA.getValue(features)
            value_prime = self.VFA.getValue(features_prime)

            # Update the VFA weights
            delta = reward + self.gamma * value_prime - value
            a = np.dot(features.T, self.auxWeights)
            delta_w = self.alpha * (features - self.gamma * features_prime) * a
            self.VFA.updateWeightsDelta(delta_w)

            # Update the GTD2 auxiliary weights
            delta_auxW = self.beta * (delta - a) * features
            self.auxWeights += delta_auxW

        return state_prime, action_prime, reward, done

# Implementation of the Gradient Q-learningincremental method for value function approximation
class GradientQlearning(AgentIncrementalVFA):
    def step(self, state, action):
        # Choose action A using a policy derived from S
        action = self.policy.getAction(self.VFA, self.featurize, state)

        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # If the agent is learning, update the VFA weights:
        if self.learn:
            # Get the action with most value
            action_star = self.policy.greedyAction(self.VFA, self.featurize, state_prime)

            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_star = self.featurize.featureStateAction(state_prime, action_star)

            # Compute the value of the features via value function approximation
            value = self.VFA.getValue(features)
            value_star = self.VFA.getValue(features_star)

            # GradientTD update step
            delta_w = self.alpha * (reward + self.gamma * value_star
                - value) * self.VFA.getGradient(features)
            self.VFA.updateWeightsDelta(delta_w)

        return state_prime, None, reward, done

# Implementation of the Recursive Least Squares TD method for value function approximation
class RLSTD(AgentIncrementalVFA):
    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        action_prime =  self.policy.getAction(self.VFA, self.featurize, state_prime)

        # If the agent is learning, update the VFA weights:
        if self.learn:
            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)

            # Compute the value of the features via function approximation
            value = self.VFA.getValue(features)
            value_prime = self.VFA.getValue(features_prime)

            g = np.matmul((features - self.gamma * features_prime).T, self.weightsRLSTD)
            a = 1 + np.dot(g, features)
            v = np.matmul(self.weightsRLSTD, features)

            # Update VFA weights
            delta = reward + self.gamma * value_prime - value
            delta_w = delta/a * v
            self.VFA.updateWeightsDelta(delta_w)

            # Update auxiliary weights
            delta_RLSTDW = - np.matmul(v, g) / a
            self.weightsRLSTD += delta_RLSTDW

        return state_prime, action_prime, reward, done

# This function demonstrates how the above methods can be used with OpenAI gym
# environments, while also demonstrating the differences in performance between
# these methods.
def compareMethods():
    import gym
    import matplotlib.pyplot as plt

    env = gym.make('GridWorld-v0')
    policy = EGreedyPolicyVFA(0.1)
    VFA = LinearVFA()
    feature = Featurize()

    training_episodes = 400
    n_plot_points = 100
    eps_benchmark = 100

    # Initialize agents
    alpha1 = 0.4
    agent1 = TD(env, policy, VFA, feature, alpha1, horizon = 20)

    alpha2 = 0.4
    lamda2 = 0.8
    agent2 = TDlamda(env, policy, VFA, feature, alpha2, lamda2, horizon = 20)

    alpha3 = 0.4
    beta3 = 0.2
    agent3 = GradientTD2(env, policy, VFA, feature, alpha3, beta = beta3, horizon = 20)

    alpha4 = 0.4
    agent4 = GradientQlearning(env, policy, VFA, feature, alpha4, horizon = 20)

    alpha5 = 0.4
    beta5 = 0.2
    agent5 = RLSTD(env, policy, VFA, feature, alpha4, beta = beta5, horizon = 20)

    agents = [agent1, agent2, agent3, agent4, agent5]

    eps_per_point = int(training_episodes / n_plot_points)
    benchmark_data = np.zeros((5, n_plot_points))
    # Benchmark agents without training
    for agent_i in range(5): benchmark_data[agent_i][0] = agents[agent_i].benchmark(eps_benchmark)
    # Train and benchmark agents
    for point_i in range(1, n_plot_points):
        for agent_i in range(5):
            print('Agent ' + str(agent_i) + ', Episode ' + str((point_i+1)*eps_per_point))
            agents[agent_i].train(eps_per_point)
            benchmark_data[agent_i][point_i] = agents[agent_i].benchmark(eps_benchmark)

    # Plot results
    xaxis = [eps_per_point*(i+1) for i in range(n_plot_points)]
    title1 = 'VFA TD, a = ' + str(alpha1)
    title2 = 'VFA TD(lamda), a = ' + str(alpha2) + ', l = ' + str(lamda2)
    title3 = 'GTD2, a = ' + str(alpha3) + ', b = ' + str(beta3)
    title4 = 'Gradient Q, a = ' + str(alpha4)
    title5 = 'RLSTD, a = ' + str(alpha5) + ', b = ' + str(beta5)
    titles = [title1, title2, title3, title4, title5]
    for i in range(5):
        plt.subplot(231+i)
        plt.plot(xaxis, benchmark_data[i])
        plt.xlabel('Training episodes')
        plt.ylabel('Average reward per episode')
        plt.title(titles[i])
    plt.show()

compareMethods()
