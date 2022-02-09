from numpy.random import choice


class Agent:
    """
    Base agent class.
    """

    def __init__(self, action_space, epsilon=1.):
        self.Q = {state: {action: 0. for action in action_space[state]} for state in action_space}
        self.epsilon = epsilon
        self.pi = self.epsilon_greedy(self.Q)

    def epsilon_greedy(self, Q):
        """Derive an epsilon-greedy policy from a given value function."""
        pi = {}
        for state in Q:
            pi[state] = {}
            epsilon_per_action = self.epsilon / len(Q[state])
            for action in Q[state]: pi[state][action] = epsilon_per_action
            pi[state][self.greedy(Q[state])] += (1 - self.epsilon)
        return pi

    def greedy(_, Q_state):
        """Return the greedy action from a dictionary of Q values for a given state."""
        return max(Q_state, key=lambda action: Q_state[action])

    def act(self, state):
        """Sample an action from the current policy."""
        return choice(list(self.pi[state].keys()), p=list(self.pi[state].values()))

    def learn(self, state, action, reward, next_state, done):
        """Not implemented for the base class."""
        pass


class QLearningAgent(Agent):
    """
    Agent class implementing the Q-learning algorithm. 
    """

    def __init__(self, action_space, epsilon, alpha, gamma):
        Agent.__init__(self, action_space, epsilon)
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor

    def learn(self, state, action, reward, next_state, done):
        """Update the value function and policy using information from the latest transition."""
        self.Q[state][action] += self.alpha * ( reward
                                              + self.gamma * self.Q[next_state][self.greedy(self.Q[next_state])]
                                              - self.Q[state][action])
        self.pi = self.epsilon_greedy(self.Q)