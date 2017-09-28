# Contains objects and functions required for the following scripts:
#   -agentTabular.py
#   -agentIncrementalVFA.py
#   -agentBatchVFA.py

import numpy as np
import pdb

# ------------------------------------------------------------------------------
# 1. Agent framework
# ------------------------------------------------------------------------------

# Contains an agent's basic functionality, such as to train and benchmark
class Agent:
    def allowlearn(self):
        self.learn = 1

    def preventlearn(self):
        self.learn = 0

    # Trains the agent by computing 'n_episodes' episodes.
    # Returns the average reward per episode.
    def train(self, n_episodes):
        if self.verbosity: print('Training...')
        self.allowlearn()
        return self.runEpisodes(n_episodes) / n_episodes

    # Benchmark the agent by computing 'n_episodes' episodes.
    # Returns the average reward per episode.
    def benchmark(self, n_episodes):
        if self.verbosity: print('benchmarking...')
        self.preventlearn()
        return self.runEpisodes(n_episodes) / n_episodes

    # Computes 'n_episodes' episodes.
    # Returns the average reward per episode.
    def runEpisodes(self, n_episodes):
        accumulatedReward = 0
        for episode_i in range(n_episodes):
            if self.verbosity: print('Episode ' + str(episode_i))
            accumulatedReward += self.episode()  # Update cumulative reward
        return accumulatedReward

# ------------------------------------------------------------------------------
# 2. Policies
# ------------------------------------------------------------------------------

# Implements an e-greedy policy for a tabular agent.
# The policy returns an action given an input state and state-action value function.
# The epsilon of the policy decays according to the parameter 'decay'
class EGreedyPolicyTabular:
    def __init__(self, epsilon, decay = 1):
        self.epsilon = epsilon
        self.decay = decay

    def getAction(self, Q, state):
        # Q(s, a) should be addressable as Q[s][a]

        if np.random.random() > self.epsilon:
            # Take greedy action
            return self.greedyAction(Q, state)
        # Take an exploratory action
        else: return self.randomAction(Q)

    # Returns a random action
    def randomAction(self, Q):
        nA = Q[0].shape[0]
        return np.random.randint(nA)

    # Returns a greedy action
    def greedyAction(self, Q, state):
        nA = Q[0].shape[0]
        maxima_index = [] # Actions with maximum value
        maxVal = None # Value of the current best actions

        for action in range(nA):
            value = Q[state][action] # Get the value from the state-action value function.
            if maxVal == None: # For the fist (s,a), intialize 'maxVal'
                maxVal = value
            if value > maxVal: # If the action is better than previus ones, update
                maxima_index = [action]
                maxVal = value
            elif value == maxVal: # If the action is equally good, add it
                maxima_index.append(action)

        # Randomly choose one of the best actions
        return np.random.choice(maxima_index)

    def epsilonDecay(self):
        self.epsilon *= self.decay

    # The policy update consists only on epsilon decay
    def episodeUpdate(self):
        self.epsilonDecay()

# Implements an e-greedy policy for an agent using value function approximation.
# The policy returns an action given an input state and VFA function.
# The epsilon of the policy decays according to the parameter 'decay'
class EGreedyPolicyVFA:
    def __init__(self, epsilon, decay = 1):
        self.epsilon = epsilon
        self.decay = decay

    def setNActions(self, nA):
        self.nA = nA

    def getAction(self, VFA, featurize, state):
        # VFA is the value function approximator
        if np.random.random() > self.epsilon:
            # Take a greedy action
            return self.greedyAction(VFA, featurize, state)
        # Take an exploratory action
        else: return self.randomAction()

    # Returns a random action
    def randomAction(self):
        return np.random.randint(self.nA)

    # Returns a greedy action
    def greedyAction(self, VFA, featurize, state):
        maxima_index = [] # Actions with maximum value
        maxVal = None # Value of the current best actions

        for action in range(self.nA):
             # Get the value of the state action pair from VFA
            features = featurize.featureStateAction(state, action)
            value = VFA.getValue(features)

            if maxVal is None: # For the fist (s,a), intialize 'maxVal'
                maxVal = value
            if value > maxVal: # If the action is better than previus ones, update
                maxima_index = [action]
                maxVal = value
            elif value == maxVal: # If the action is equally good, add it
                maxima_index.append(action)

        # Randomly choose one of the best actions
        return np.random.choice(maxima_index)

    # Returns an array containing the action with the highest value for every state
    def getDetArray(self, VFA, featurize, nS):
        detActions = np.zeros((nS, 1))
        actionVals = np.zeros((self.nA, 1)) # Stores the values for all actions
                                            # in a given state
        for state in range(nS):
            for action in range(self.nA):
                features = featurize.featureStateAction(state, action)
                actionVals[action] = VFA.getValue(features)
            detActions[state] = np.argmax(actionVals) # Choose highest value
        return detActions

    def epsilonDecay(self):
        self.epsilon *= self.decay

    # The policy update consists only on epsilon decay
    def episodeUpdate(self):
        self.epsilonDecay()

# Implements a softmax policy for an agent using linear combination of features
# for value function approximation.
# The policy returns an action given an input state and VFA function.
class SoftmaxPolicyVFA:
    def __init__(self, tau = 1):
        self.tau = tau

    # Intialize the weights vector to a fixed  value
    def setUpWeights(self, dimensions, value = 1):
        self.weights = np.ones(dimensions) * value

    def setNActions(self, nA):
        self.nA = nA

    def getAction(self, featurize, state):
        probabilities = self.computeWeights(featurize, state)

        # Sample action according to the computed probabilities
        return np.random.choice(range(self.nA), p = probabilities)

    # Returns a greedy action
    def greedyAction(self, featurize, state):
        probabilities = self.computeWeights(VFA, featurize, state)

        # Return action with the highest probability
        return np.argmax(probabilities)

    # Compute the probability of sampling each action in a softmax maner
    def computeWeights(self, featurize, state):
        # Compute the feature vector
        values = np.zeros((self.nA, 1))
        for action in range(self.nA):
            feature = featurize.featureStateAction(state, action)
            values[action] = np.dot(feature.T, self.weights)

        # Get the weight of each action
        values_exp = np.exp(values / self.tau - max(values))
        probabilities = (values_exp / sum(values_exp)).flatten()
        #print probabilities
        return probabilities

    # Compute the policy gradient for the state-action pair
    def getGradient(self, featurize, state, action):
        # Compute the feature for every action
        features = featurize.featureStateAction(state, 0) # Array to store the features
        for a in range(1, self.nA): features = np.hstack([features, featurize.featureStateAction(state, a)])
        mean_feature = np.mean(features, 1).reshape(-1,1)  # Mean of the features
        gradient = (features[:, action].reshape(-1, 1) - mean_feature) / self.tau  # Compute gradient
        return gradient

    # Update the parameter theta
    def updateWeightsDelta(self, delta):
        self.weights += delta

# ------------------------------------------------------------------------------
# 3. Featurize states
# ------------------------------------------------------------------------------

# General object to featurize states:
#   -featureState(state) returns the feature corresponding to state s
#   -featureStateAction(state, action) returns the feature corresponding to the
#       state-action pair (s,a)
class Featurize():
    def set_nSnA(self, nS, nA):
        self.nS = nS
        self.nA = nA

    def featureState(self, state):
        return featureTableState(state, self.nS)

    def featureStateAction(self, state, action):
        return featureTableStateAction(state, action, self.nS, self.nA)

# Specific implementation of feature functions

# 1. Table lookup
# Function to featurize a state using table lookup
def featureTableState(state, nS):
    feature = np.zeros((nS, 1))
    feature[state] = 1
    return feature
# Function to featurize a state-action pair using table lookup
def featureTableStateAction(state, action, nS, nA):
    feature = np.zeros((nS * nA, 1))
    feature[state * nA + action] = 1
    return feature

# ------------------------------------------------------------------------------
# 4. Value function approximation
# ------------------------------------------------------------------------------

# Implementation of linear value function approximation through linear
# combination of features
class LinearVFA:
    # Intialize the weights vector to a fixed  value
    def setUpWeights(self, dimensions, value = 1):
        self.weights = np.ones(dimensions) * value

    def returnWeights(self, dimensions, value = 1):
        return np.ones(dimensions) * value

    def getValue(self, features):
        return np.dot(features.T, self.weights)

    def getGradient(self, features):
        return features

    def updateWeightsDelta(self, delta_weight):
        self.weights += delta_weight

    def updateWeightsMatrix(self, A, b):
        self.weights = np.matmul(np.linalg.inv(A), b)

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights

# ------------------------------------------------------------------------------
# 5. Models
# ------------------------------------------------------------------------------

# Implementation of a Table Lookup Model as showed by David Silver in
# COMPM050/COMPGI13 Lecture 8, slide 15
class TableLookupModel:
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.N = np.zeros((nS, nA)) # Keep track of the number of times (s,a)
                                    # has appeared
        self.SprimeCounter = np.zeros((nS, nA, nS)) # Number of times (s,a)
                                                    # resulted in s'
        self.Rcounter = np.zeros((nS, nA)) # Total reward obtained by (s,a)
        self.observedStates = [] # states that have appeared before
        self.observedActions = [[] for i in range(nS)] # actions observed before
                                                       # at every state
        self.terminalStates = [] # No knowledge about terminal states assumed

    # Experience is considered as a tuple of (state, action, reward, state_prime)
    def addExperience(self, experience):
        s, a, r, s_prime = experience
        self.N[s][a] += 1
        self.SprimeCounter[s][a][s_prime] += 1
        self.Rcounter[s][a] += r
        if not s in self.observedStates: self.observedStates.append(s)
        if not a in self.observedActions[s]: self.observedActions[s].append(a)

    # Samples the resulting state of (s,a)
    def sampleStatePrime(self, state, action):
        # If there is no information about (s,a), then sample randomly
        if self.N[state][action] == 0: return np.random.choice(range(self.nS))

        prob = self.SprimeCounter[state][action] / self.N[state][action]
        return np.random.choice(range(self.nS), p = prob)

    # Samples the resulting reward of (s,a)
    def sampleReward(self, state, action):
        # If there is no information about (s,a), then return a fixed reward
        if self.N[state][action] == 0: return 0

        return self.Rcounter[state][action] / self.N[state][action]

    # Sample a random state that has been observed before
    def sampleRandState(self):
        return np.random.choice(self.observedStates)

    # Sample a random action previously observed in a given state
    def sampleRandAction(self, state):
        return np.random.choice(self.observedActions[state])

    # Give model knowledge about terminal states
    def addTerminalStates(self, term_states):
        self.terminalStates = term_states

    # Check wether a state is terminal (assuming model has knowledge about
    # terminal states)
    def isTerminal(self, state):
        return state in self.terminalStates
