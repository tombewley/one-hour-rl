from numpy.random import choice


class Agent:
    """
    Base agent class.
    """

    def __init__(self, mdp, epsilon=1.):
        self.Q = {state: {action: 0. for action in spec["actions"]} for state, spec in mdp._spec.items()}
        self.epsilon = epsilon
        self.policy = self.epsilon_greedy(self.Q)

    def epsilon_greedy(self, Q):
        """Derive an epsilon-greedy policy from a given value function."""
        policy = {}
        for state in Q:
            policy[state] = {}
            num_actions = len(Q[state])
            if num_actions > 0:
                epsilon_per_action = self.epsilon / num_actions
                for action in Q[state]: policy[state][action] = epsilon_per_action
                policy[state][self.greedy(Q[state])] += (1 - self.epsilon)
        return policy

    def greedy(_, Q_state):
        """Return the greedy action from a dictionary of Q values for a given state."""
        return max(Q_state, key=lambda action: Q_state[action])

    def act(self, state):
        """Sample an action from the current policy."""
        return choice(list(self.policy[state].keys()), p=list(self.policy[state].values()))

    def learn(self, state, action, reward, next_state, done):
        """Not implemented for the base class."""
        pass


class QLearningAgent(Agent):
    """
    Agent class implementing the Q-learning algorithm. 
    """

    def __init__(self, mdp, epsilon, alpha, gamma):
        super(QLearningAgent, self).__init__(mdp, epsilon) 
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor

    def learn(self, state, action, reward, next_state, done):
        """Update the value function and policy using information from the latest transition."""
        Q_next = 0. if done else self.Q[next_state][self.greedy(self.Q[next_state])]
        self.Q[state][action] += self.alpha * ( reward + self.gamma * Q_next - self.Q[state][action])
        self.policy = self.epsilon_greedy(self.Q) # NOTE: A little inefficient as it rebuilds the dictionaries from scratch