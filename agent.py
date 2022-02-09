from numpy.random import choice


class Agent:
    """
    Base agent class.
    """

    def __init__(self, action_space):
        self.Q, self.pi = {}, {}
        for state in action_space:
            self.Q[state], self.pi[state] = {}, {}
            n = len(action_space[state])
            for action in action_space[state]:
                self.Q[state][action] = 0.
                self.pi[state][action] = 1 / n # Uniform random policy

    def act(self, state):
        """Sample an action from the current policy."""
        return choice(list(self.pi[state].keys()), p=list(self.pi[state].values()))

    def learn(self, state, action, reward, next_state, done):
        """Update the policy using information from the latest transition."""
        pass