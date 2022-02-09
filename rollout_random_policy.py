from mdp import StudentMDP


mdp = StudentMDP()

for ep in range(1):
    print("="*27+f" EPISODE {str(ep).rjust(3)} "+"="*27)
    state = mdp.reset()
    done = False
    while not done:
        action = mdp.random_action()
        state, reward, done, info = mdp.step(action, verbose=True)