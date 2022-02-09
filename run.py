from mdp import StudentMDP
from agent import Agent


mdp = StudentMDP(verbose=True)
agent = Agent(mdp.action_space)

while mdp.ep < 10:
    state = mdp.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = mdp.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state