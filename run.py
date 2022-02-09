from mdp import StudentMDP
from agent import Agent, QLearningAgent


mdp = StudentMDP(verbose=True)

if True:
    # Default policy shown in images/student-mdp.png
    agent = Agent(mdp.action_space) 
    agent.pi = {
        "Class 1":  {"Study": 0.5, "Go on Facebook": 0.5},
        "Class 2":  {"Study": 0.8, "Fall asleep": 0.2},
        "Class 3":  {"Study": 0.6, "Go to the pub": 0.4},
        "Facebook": {"Keep scrolling": 0.9, "Close Facebook": 0.1},
        "Pub":      {"Have a pint": 1.},
        "Pass":     {"Fall asleep": 1.},
        "Asleep":   {"Stay asleep": 1.}
    }
else:
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