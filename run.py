from mdp import StudentMDP
from agent import Agent
from q_learning_agent import QLearningAgent


mdp = StudentMDP(verbose=True)

# agent = Agent(mdp.action_space) # Random agent
agent = QLearningAgent(mdp.action_space, epsilon=0.5, alpha=0.2, gamma=0.9)

while mdp.ep < 100:
    state = mdp.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = mdp.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state

    print(agent.Q)