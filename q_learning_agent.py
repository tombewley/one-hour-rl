from agent import Agent


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