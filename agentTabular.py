# Implementation of the following tabular action value reinforcement learning methods:
#       - SARSA [1]
#       - SARSA(lamda) [2]
#       - Q-learning [3]
#       - Watkins Q-learning(lamda) [4]
# to be used with OpenAI Gym environments. Demonstrations are included with the
# following environments: FrozenLake-v0.
#
# [1] - Sutton and Barto (2012), Reinforcement Learning: an Introduction, page 142
# [2] - Sutton and Barto (2012), Reinforcement Learning: an Introduction, page 171
# [3] - Sutton and Barto (2012), Reinforcement Learning: an Introduction, page 145
# [4] - Sutton and Barto (2012), Reinforcement Learning: an Introduction, page 174
#
# By Ricardo Dominguez Olmedo, Aug-2017

# Import necessary libraries
import numpy as np
import gym
import matplotlib.pyplot as plt

# Contains the agent's basic functionality, such as to train and benchmark the agent.
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

    # benchmark_data the agent by computing 'n_episodes' episodes.
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

# Implements the specific functionality of a tabular agent, such as to initilize
# the agent or run episodes.
class TabularAgent(Agent):
    def __init__(self, env, policy, alpha, lamda = 0, gamma = 1, fixedQval = 0,
        horizon = 1000, verbosity = 0):
        # Inputs:
        #   -env: openAI gym environment object
        #   -alpha: step size parameter for value function update
        #   -lamda: trace discount paramater
        #   -gamma: reward discount-rate parameter
        #   -fixedQval: initial value for all states and actions of the
        #       state-action value function
        #   -horizon: finite horizon steps
        #   -verbosity: if TRUE, prints to screen additional information

        self.env = env
        self.policy = policy
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma
        self.horizon = horizon
        self.verbosity = verbosity

        self.nS = env.observation_space.n   # Number of states
        self.nA = env.action_space.n        # Number of actions

        # Initialize the state-action value function
        self.Q = np.ones((self.nS, self.nA)) * fixedQval

        # Initially prevent agent from learning
        self.learn = 0

    # Computes a single episode.
    # Returns the episode reward return.
    def episode(self):
        episodeReward = 0
        self.setUpTrace()

        # Initialize S, A
        state = self.env.reset()
        action = self.policy.getAction(self.Q, state) # Choose action

        # Repeat for each episode
        for t in range(self.horizon):
            # Take action A, observe R, S'
            state, action, reward, done = self.step(state, action)

            # Update the total episode return
            episodeReward += reward

            # Finish loop if S' is a terminal state
            if done: break

        # Update the policy parameters if the agent is learning
        if self.learn: self.policy.episodeUpdate()

        return episodeReward

    # Initilize trace matrix
    def setUpTrace(self):
        self.E = np.zeros((self.nS, self.nA))

# Implements an e-greedy policy for a tabular agent.
# The policy returns an action given an input state and state-action valueu function.
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


# Implementation of the SARSA tabular action value method
class SARSA(TabularAgent):
    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from Q(s,a) and S'
        action_prime = self.policy.getAction(self.Q, state_prime)

        # If the agent is learning, update Q(s,a) using TD(0)
        if self.learn:
            self.Q[state][action] += self.alpha * (reward + self.gamma *
                self.Q[state_prime][action_prime] - self.Q[state][action])

        return state_prime, action_prime, reward, done

# Implementation of the SARSA(lamda) tabular action value method
class SARSAlamda(TabularAgent):
    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from Q(s,a) and S'
        action_prime = self.policy.getAction(self.Q, state_prime)

        # If the agent is learning, update Q(s,a) using TD(lamda)
        if self.learn:
            td_error = (reward + self.gamma * self.Q[state_prime][action_prime]
                - self.Q[state][action])

            self.E[state][action] += 1  # Update current state trace
            for state_i in range(self.nS):
                for action_i in range(self.nA):
                    self.Q[state_i][action_i] += self.alpha * td_error * self.E[state_i][action_i]
                    self.E[state_i][action_i] *= self.gamma * self.lamda # Update traces

        return state_prime, action_prime, reward, done

# Implementation of the Q-learning tabular action value method
class Qlearning(TabularAgent):
    def step(self, state, action):
        # Choose A using a policy derived from Q(s,a) and S
        action = self.policy.getAction(self.Q, state)

        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # If the agent is learning, update Q(s,a) using the maximum value from
        # S' as the TD update target
        if self.learn:
            td_target = reward + self.gamma * np.max(self.Q[state_prime])
            self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])

        return state_prime, None, reward, done

# Implementation of Watkins Q-learning(lamda) tabular action value method
class QlearWatkins(TabularAgent):
    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from Q(s,a) and S'
        action_prime = self.policy.getAction(self.Q, state_prime)

        if self.learn:
        # If the agent is learning, update Q(s,a)
            max_Qvalue = np.max(self.Q[state_prime])
            td_error = reward + self.gamma * max_Qvalue - self.Q[state][action]

            # Check wether the action choosen was exploratory or not
            exploratory_action = 0 if max_Qvalue == self.Q[state_prime][action] else 1

            self.E[state][action] += 1 # Update trace for the current state
            for state_i in range(self.nS):
                for action_i in range(self.nA):
                    self.Q[state_i][action_i] += self.alpha * td_error * self.E[state_i][action_i]

                    # Update traces
                    if exploratory_action: self.E[state_i][action_i] = 0
                    else: self.E[state_i][action_i] *= self.gamma * self.lamda

        return state_prime, action_prime, reward, done

# This function demonstrates how the above methods can be used with OpenAI gym
# environments, while also demonstrating the differences in performance between
# these methods.
def compareMethods():
    env = gym.make('GridWorld-v0')
    policy = EGreedyPolicyTabular(0.1)

    training_episodes = 400
    n_plot_points = 100
    eps_benchmark = 100

    # Initialize agents
    alpha1 = 0.4
    agent1 = SARSA(env, policy, alpha1, horizon = 20)

    alpha2 = 0.4
    lamda2 = 0.8
    agent2 = SARSAlamda(env, policy, alpha2, lamda2, horizon = 20)

    alpha3 = 0.4
    agent3 = Qlearning(env, policy, alpha3, horizon = 20)

    alpha4 = 0.4
    lamda4 = 0.8
    agent4 = QlearWatkins(env, policy, alpha4, lamda4, horizon = 20)

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
    plt.figure(figsize=(13, 10))
    xaxis = [eps_per_point*(i+1) for i in range(n_plot_points)]
    title1 = 'SARSA, a = ' + str(alpha1)
    title2 = 'SARSA(lamda), a = ' + str(alpha2) + ', l = ' + str(lamda2)
    title3 = 'Q-learning, a = ' + str(alpha3)
    title4 = 'Watkins Q, a = ' + str(alpha4) + ', l = ' + str(lamda4)
    titles = [title1, title2, title3, title4]
    for i in range(4):
        plt.subplot(221+i)
        plt.plot(xaxis, benchmark_data[i])
        plt.xlabel('Training episodes')
        plt.ylabel('Average reward per episode')
        plt.title(titles[i])
    plt.show()

compareMethods()
