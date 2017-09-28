# Implementation of the following actor critic methods:
#   - Q Actor-Critic [1]
#   - Advantage Actor-Critic [2]
#   - TD Actor-Critic [2]
#   - TD(lamda) Actor-Critic [3]
# to be used with OpenAI Gym environments. Demonstrations are included with the
# following environments: GridWorld-v0.
#
# [1] - David Silver (2015), COMPM050/COMPGI13 Lecture 7, slide 25
# [2] - David Silver (2015), COMPM050/COMPGI13 Lecture 7, slide 31
# [4] - David Silver (2015), COMPM050/COMPGI13 Lecture 7, slide 34
#
# By Ricardo Dominguez Olmedo, Aug-2017

# Import necessary libraries and functions
import numpy as np
from util import Agent
from util import Featurize
from util import LinearVFA
from util import SoftmaxPolicyVFA

class AgentQAC(Agent):
    def __init__(self, env, policy, VFAcritic, featurize, alpha, beta, lamda = 0,
    gamma = 1, horizon = 1000, verbosity = 0):
        # Inputs:
        #   -env: openAI gym environment object
        #   -policy: object containing a policy from which to sample actions
        #   -VFA: object containing the value function approximator
        #   -featurize: object which featurizes states
        #   -alpha: step size parameter
        #   -beta: secondary step size parameter
        #   -kappa: step size parameter for state value function (AAC, TDAC)
        #   -lamda: trace discount paramater
        #   -gamma: discount-rate parameter
        #   -horizon: finite horizon steps
        #   -verbosity: if TRUE, prints to screen additional information

        self.env = env
        self.policy = policy
        self.featurize = featurize
        self.VFAcritic = VFAcritic
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.gamma = gamma
        self.horizon = horizon
        self.verbosity = verbosity

        self.nS = env.observation_space.n   # Number of states
        self.nA = env.action_space.n    # Number of actions
        self.policy.setNActions(self.nA)
        self.featurize.set_nSnA(self.nS, self.nA)
        self.featDim = featurize.featureStateAction(0,0).shape # Dimensions of the
                                                               # feature vector
        self.policy.setUpWeights(self.featDim)
        self.setUpCritic(self.featDim, self.nS, self.nA)

        # Initially prevent agent from learning
        self.learn = 0

    def setUpTrace(self):
        self.E = np.zeros(self.featDim)

    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        action_prime = self.policy.getAction(self.featurize, state_prime)

        # Store experience
        if self.learn:
            # If traces are being used, update them
            if self.lamda != 0:
                features = self.featurize.featureStateAction(state, action)
                self.E = (self.gamma * self.lamda * self.E) + self.VFA.getGradient(features)

            # Store experience
            self.sequence.append((state, action, reward, state_prime, action_prime, self.E))

        return state_prime, action_prime, reward, done

    # Computes a single episode.
    # Returns the episode reward return.
    def episode(self):
        episodeReward = 0
        self.setUpTrace()

        # Initialize S, A
        state = self.env.reset()
        action = self.policy.getAction(self.featurize, state)

        # Repeat for each episode
        for t in range(self.horizon):
            # Take action A, observe R, S'
            state, action, reward, done = self.step(state, action)

            # Update the total episode return
            episodeReward += reward

            # Finish the loop if S' is a terminal state
            if done: break

        return episodeReward

class QAC(AgentQAC):
    def setUpCritic(self, featDim, nS, nA):
        self.VFAcritic.setUpWeights(featDim) # Initialize weights critic VFA

    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        action_prime = self.policy.getAction(self.featurize, state_prime)

        if self.learn:
            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)

            # Compute the value of the features via value function approximation
            value = self.VFAcritic.getValue(features)
            value_prime = self.VFAcritic.getValue(features_prime)
            delta = reward + self.gamma * value_prime - value

            # Actor update
            gradient = self.policy.getGradient(self.featurize, state, action)
            delta_theta = self.alpha * gradient * value
            self.policy.updateWeightsDelta(delta_theta)

            # Critic update
            delta_weight = self.beta * delta * features
            self.VFAcritic.updateWeightsDelta(delta_weight)

        return state_prime, action_prime, reward, done

class AdvanAC(AgentQAC):
    def setUpCritic(self, featDim, nS, nA):
        self.VFAcritic.setUpWeights(featDim) # Initialize weights critic VFA

        # In order to compute the advantage function it is necesary to have
        # another set of weights to approximate both Q(s,a) and V(s)
        # The VFA chosen here is linear combination of features
        self.VFAstateval = LinearVFA()
        self.VFAstateval.setUpWeights((self.nS, 1))
        self.kappa = self.beta * 0.4 # Step size parameter

    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        action_prime = self.policy.getAction(self.featurize, state_prime)

        if self.learn:
            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)
            features_state = self.featurize.featureState(state)
            features_stateprime = self.featurize.featureState(state_prime)

            # Compute the value of the features via value function approximation
            value = self.VFAcritic.getValue(features)
            value_prime = self.VFAcritic.getValue(features_prime)
            value_state = self.VFAstateval.getValue(features_state)
            value_stateprime = self.VFAstateval.getValue(features_stateprime)

            delta_q = reward + self.gamma * value_prime - value
            delta_v = reward + self.gamma * value_stateprime - value_state

            # Actor update
            advantage = value - value_state
            gradient = self.policy.getGradient(self.featurize, state, action)
            delta_theta = self.alpha * gradient * advantage
            self.policy.updateWeightsDelta(delta_theta)

            # Critic update
            delta_weight = self.beta * delta_q * features
            self.VFAcritic.updateWeightsDelta(delta_weight)

            # State value function update
            delta_stateWeight = self.kappa * delta_v * features_state
            self.VFAstateval.updateWeightsDelta(delta_stateWeight)

        return state_prime, action_prime, reward, done

class TDAC(AgentQAC):
    def setUpCritic(self, featDim, nS, nA):
        self.VFAcritic.setUpWeights((self.nS, 1)) # Initialize weights critic VFA

    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        action_prime = self.policy.getAction(self.featurize, state_prime)

        if self.learn:
            # Compute the pertinent feature vectors
            features = self.featurize.featureState(state)
            features_prime = self.featurize.featureState(state_prime)

            # Compute the value of the features via value function approximation
            value = self.VFAcritic.getValue(features)
            value_prime = self.VFAcritic.getValue(features_prime)
            delta = reward + self.gamma * value_prime - value

            # Actor update
            gradient = self.policy.getGradient(self.featurize, state, action)
            delta_theta = self.alpha * gradient * delta
            self.policy.updateWeightsDelta(delta_theta)

            # Critic update
            delta_weight = self.beta * delta * features
            self.VFAcritic.updateWeightsDelta(delta_weight)

        return state_prime, action_prime, reward, done

class TDlamdaAC(AgentQAC):
    def setUpCritic(self, featDim, nS, nA):
        self.VFAcritic.setUpWeights((self.nS, 1)) # Initialize weights critic VFA

    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        action_prime = self.policy.getAction(self.featurize, state_prime)

        if self.learn:
            # Compute the pertinent feature vectors
            features = self.featurize.featureState(state)
            features_prime = self.featurize.featureState(state_prime)

            # Compute the value of the features via value function approximation
            value = self.VFAcritic.getValue(features)
            value_prime = self.VFAcritic.getValue(features_prime)
            delta = reward + self.gamma * value_prime - value

            # Actor update
            delta_theta = self.alpha * delta * self.E
            self.policy.updateWeightsDelta(delta_theta)

            # Trace update
            gradient = self.policy.getGradient(self.featurize, state, action)
            self.E = self.lamda * self.E + gradient

            # Critic update
            delta_weight = self.beta * delta * features
            self.VFAcritic.updateWeightsDelta(delta_weight)

        return state_prime, action_prime, reward, done

class NaturalAC(AgentQAC):
    def setUpCritic(self, featDim, nS, nA):
        self.VFAcritic.setUpWeights(featDim) # Initialize weights critic VFA

    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        action_prime = self.policy.getAction(self.featurize, state_prime)

        if self.learn:
            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)

            # Compute the value of the features via value function approximation
            value = self.VFAcritic.getValue(features)
            value_prime = self.VFAcritic.getValue(features_prime)
            delta = reward + self.gamma * value_prime - value

            # Actor update
            gradient = self.policy.getGradient(self.featurize, state, action)
            delta_theta = self.alpha * gradient * self.VFAcritic.getWeights()
            self.policy.updateWeightsDelta(delta_theta)

            # Critic update
            delta_weight = self.beta * delta * features
            self.VFAcritic.updateWeightsDelta(delta_weight)

        return state_prime, action_prime, reward, done
# This function demonstrates how the above methods can be used with OpenAI gym
# environments, while also demonstrating the differences in performance between
# these methods.
def compareMethods():
    import gym
    import matplotlib.pyplot as plt

    env = gym.make('GridWorld-v0')
    policy = SoftmaxPolicyVFA(1)
    feature = Featurize()

    training_episodes = 1000
    n_plot_points = 100
    eps_benchmark = 100
    fixedHorizon = 20

    agent = AdvanAC(env, policy, LinearVFA(), feature, 0.2, 0.1, 0.4, horizon = 20)

    # Initialize agents
    alpha1 = 0.2
    beta1 = 0.1
    agent1 = QAC(env, policy, LinearVFA(), feature, alpha1, beta1, horizon = fixedHorizon)

    alpha2 = 0.2
    beta2 = 0.1
    agent2 = AdvanAC(env, policy, LinearVFA(), feature, alpha2, beta2, horizon = fixedHorizon)

    alpha3 = 0.2
    beta3 = 0.1
    agent3 = TDAC(env, policy, LinearVFA(), feature, alpha3, beta3, horizon = fixedHorizon)

    alpha4 = 0.2
    beta4 = 0.1
    lamda4 = 0.4
    agent4 = TDlamdaAC(env, policy, LinearVFA(), feature, alpha4, beta4, lamda4, horizon = fixedHorizon)

    alpha5 = 0.2
    beta5 = 0.1
    agent5 = NaturalAC(env, policy, LinearVFA(), feature, alpha5, beta5, horizon = fixedHorizon)
    agents = [agent1, agent2, agent3, agent4, agent5]

    eps_per_point = int(training_episodes / n_plot_points)
    benchmark_data = np.zeros((5, n_plot_points))
    # Benchmark agents without training
    for agent_i in range(5): benchmark_data[agent_i][0] = agents[agent_i].benchmark(eps_benchmark)
    # Train and benchmark agents
    for point_i in range(1, n_plot_points):
        for agent_i in range(5):
            print('Agent ' + str(agent_i+1) + ', Episode ' + str((point_i+1)*eps_per_point))
            agents[agent_i].train(eps_per_point)
            benchmark_data[agent_i][point_i] = agents[agent_i].benchmark(eps_benchmark)

    # Plot results
    plt.figure(figsize=(16, 10))
    xaxis = [eps_per_point*(i+1) for i in range(n_plot_points)]
    title1 = 'QAC, a = ' + str(alpha1) + ' b = ' + str(beta1)
    title2 = 'Advantage AC, a = ' + str(alpha2) + ' b = ' + str(beta2)
    title3 = 'TDAC, a = ' + str(alpha3) + ' b = ' + str(beta3)
    title4 = 'TD(lamda)AC, a = ' + str(alpha4) + ' b = ' + str(beta4) + ' l =' + str(lamda4)
    title5 = 'Natural AC, a = ' + str(alpha5) + ' b = ' + str(beta5)
    titles = [title1, title2, title3, title4, title5]
    for i in range(5):
        plt.subplot(231+i)
        plt.plot(xaxis, benchmark_data[i])
        plt.xlabel('Training episodes')
        plt.ylabel('Average reward per episode')
        plt.title(titles[i])
    plt.show()

compareMethods()
