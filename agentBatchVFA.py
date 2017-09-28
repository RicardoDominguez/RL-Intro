# Implementation of the following batch methods for value function approximation
# with policy iteration, using linear combination of features and table lookup features:
#   - Least Squares TD(0) [1]
#   - Least Squares TD(lamda) [1]
#   - Least Squares TDQ [2]
#   - Least Squares Policy Iteration TD [3]
# to be used with OpenAI Gym environments. Demonstrations are included with the
# following environments: GridWorld-v0.
#
# The control implementation for this batch methods are not efficient, but rather
# demonstrate their ability to be used for function value evaluation given some
# training experience.
#
# [1] - David Silver (2015), COMPM050/COMPGI13 Lecture 6, slide 45
# [2] - David Silver (2015), COMPM050/COMPGI13 Lecture 6, slide 50
# [3] - David Silver (2015), COMPM050/COMPGI13 Lecture 6, slide 51
#
# By Ricardo Dominguez Olmedo, Aug-2017

# Import necessary libraries and functions
import numpy as np
from util import Agent
from util import Featurize
from util import LinearVFA
from util import EGreedyPolicyVFA

# Implements the specific functionality of a batch value function approximation
# agent, such as to initialize the agent or run episodes.
class BatchAgent(Agent):
    def __init__(self, env, policy, VFA, featurize, alpha, batchSize = 100,
        lamda = 0, gamma = 1, eps = 1, horizon = 1000, verbosity = 0):
        # Inputs:
        #   -env: openAI gym environment object
        #   -policy: object containing a policy from which to sample actions
        #   -VFA: object containing the value function approximator
        #   -featurize: object which featurizes states
        #   -alpha: step size parameter
        #   -batchSize: number of episodes of experience before policy evaluation
        #   -lamda: trace discount paramater
        #   -gamma: discount-rate parameter
        #   -eps: minimum difference in a weight update for methods that require
        #       convergence
        #   -horizon: finite horizon steps
        #   -verbosity: if TRUE, prints to screen additional information

        self.env = env
        self.policy = policy
        self.featurize = featurize
        self.VFA = VFA
        self.alpha = alpha
        self.batchSize = batchSize
        self.lamda = lamda
        self.gamma = gamma
        self.eps = eps
        self.horizon = horizon
        self.verbosity = verbosity

        self.nS = env.observation_space.n   # Number of states
        self.nA = env.action_space.n    # Number of actions
        self.policy.setNActions(self.nA)
        self.featurize.set_nSnA(self.nS, self.nA)
        self.featDim = featurize.featureStateAction(0,0).shape # Dimensions of the
                                                               # feature vector
        self.VFA.setUpWeights(self.featDim) # Initialize weights for the VFA
        self.learn = 0 # Initially prevent agent from learning

        self.batch_i = 0 # To keep track of the number of stored experience episodes
        self.sequence =  [] # Array to store episode sequences

    def setUpTrace(self):
        self.E = np.zeros(self.featDim)

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

        # Update the policy if the agent is learning and the amount of required
        # experience is met.
        if self.learn:
             self.batch_i += 1
             if (self.batch_i+1) % self.batchSize == 0: self.batchUpdate()

        return episodeReward

    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        action_prime = self.policy.getAction(self.VFA, self.featurize, state_prime)

        # Store experience
        if self.learn:
            # If traces are being used, update them
            if self.lamda != 0:
                features = self.featurize.featureStateAction(state, action)
                self.E = (self.gamma * self.lamda * self.E) + self.VFA.getGradient(features)

            # Store experience
            self.sequence.append((state, action, reward, state_prime, action_prime, self.E))

        return state_prime, action_prime, reward, done

# Implementation of the Linear Least Squares TD batch prediction algorithm
class LeastSquaresTD(BatchAgent):
    def batchUpdate(self):
        A = np.zeros((self.nA * self.nS, self.nA * self.nS))
        b = np.zeros((self.nA * self.nS, 1))
        for di, dn in enumerate(self.sequence):
            # Get data from array
            state, action, reward, state_prime, action_prime, E = dn

            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)

            A_delta = np.matmul(features, (features - self.gamma * features_prime).T)
            A += A_delta

            b_delta = reward * features
            b += b_delta

        if np.linalg.det(A) != 0: self.VFA.updateWeightsMatrix(A, b)

# Implementation of the Linear Least Squares TD batch prediction algorithm using
# eligibility traces.
class LSTDlamda(BatchAgent):
    def batchUpdate(self):
        A = np.zeros((self.nA * self.nS, self.nA * self.nS))
        b = np.zeros((self.nA * self.nS, 1))
        for di, dn in enumerate(self.sequence):
            # Get data from array
            state, action, reward, state_prime, action_prime, E = dn

            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)

            A_delta = np.matmul(E, (features - self.gamma * features_prime).T)
            A += A_delta

            b_delta = reward * E
            b += b_delta

        if np.linalg.det(A) != 0: self.VFA.updateWeightsMatrix(A, b)

# Implementation of the Linear Least Squares TDQ batch prediction algorithm
class LSTDQ(BatchAgent):
    def batchUpdate(self):
        A = np.zeros((self.nA * self.nS, self.nA * self.nS))
        b = np.zeros((self.nA * self.nS, 1))
        for di, dn in enumerate(self.sequence):
            # Get data from array
            state, action, reward, state_prime, action_prime, E = dn

            # Compute A' greedily from S'
            action_star = self.policy.greedyAction(self.VFA, self.featurize, state_prime)

            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_star)

            A_delta = np.matmul(features, (features - self.gamma * features_prime).T)
            A += A_delta

            b_delta = reward * features
            b += b_delta

        if np.linalg.det(A) != 0: self.VFA.updateWeightsMatrix(A, b)

# Implementation of the Linear Least Squares Policy Iteration with LSTDQ
# batch evaluation method
class LSPITD(BatchAgent):
    def batchUpdate(self):
        pi_prime = self.policy.getDetArray(self.VFA, self.featurize, self.nS)
        while 1:
            pi = pi_prime
            self.updateWeights()
            pi_prime = self.policy.getDetArray(self.VFA, self.featurize, self.nS)
            if np.array_equal(pi, pi_prime): break

    def updateWeights(self):
        A = np.zeros((self.nA * self.nS, self.nA * self.nS))
        b = np.zeros((self.nA * self.nS, 1))
        for di, dn in enumerate(self.sequence):
            # Get data from array
            state, action, reward, state_prime, action_prime, E = dn

            # Compute A' greedily from S'
            action_star = self.policy.greedyAction(self.VFA, self.featurize, state_prime)

            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_star)

            A_delta = np.matmul(features, (features - self.gamma * features_prime).T)
            A += A_delta

            b_delta = reward * features
            b += b_delta

        if np.linalg.det(A) != 0: self.VFA.updateWeightsMatrix(A, b)

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

    training_episodes = 1000
    n_plot_points = 100
    eps_benchmark = 100
    fixedHorizon = 20

    # Initialize agents
    alpha1 = 0.4
    agent1 = LeastSquaresTD(env, policy, VFA, feature, alpha1, horizon = fixedHorizon)

    alpha2 = 0.4
    lamda2 = 0.8
    agent2 = LSTDlamda(env, policy, VFA, feature, alpha2, lamda2, horizon = fixedHorizon)

    alpha3 = 0.4
    agent3 = LSTDQ(env, policy, VFA, feature, alpha3, horizon = fixedHorizon)

    alpha4 = 0.4
    agent4 = LSPITD(env, policy, VFA, feature, alpha4, horizon = fixedHorizon)

    agents = [agent1, agent2, agent3, agent4]

    eps_per_point = int(training_episodes / n_plot_points)
    benchmark_data = np.zeros((4, n_plot_points))
    # Benchmark agents without training
    for agent_i in range(4): benchmark_data[agent_i][0] = agents[agent_i].benchmark(eps_benchmark)
    # Train and benchmark agents
    for point_i in range(1, n_plot_points):
        for agent_i in range(4):
            print('Agent ' + str(agent_i) + ', Episode ' + str((point_i+1)*eps_per_point))
            agents[agent_i].train(eps_per_point)
            benchmark_data[agent_i][point_i] = agents[agent_i].benchmark(eps_benchmark)

    # Plot results
    plt.figure(figsize=(12, 10))
    xaxis = [eps_per_point*(i+1) for i in range(n_plot_points)]
    title1 = 'LSTD(0), a = ' + str(alpha1)
    title2 = 'LSTD(lamda), a = ' + str(alpha2) + ', l = ' + str(lamda2)
    title3 = 'LSTDQ, a = ' + str(alpha3)
    title4 = 'LSPITD, a = ' + str(alpha4)
    titles = [title1, title2, title3, title4]
    for i in range(4):
        plt.subplot(221+i)
        plt.plot(xaxis, benchmark_data[i])
        plt.xlabel('Training episodes')
        plt.ylabel('Average reward per episode')
        plt.title(titles[i])
    plt.show()

compareMethods()
