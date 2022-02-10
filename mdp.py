from numpy.random import choice


class StudentMDP:
    """
    OpenAI Gym-compatible class for the Student MDP from David Silver's "Introduction to Reinforcement Learning" class.
    See images/student-mdp.png for a diagram of this MDP.

    Code adapted from https://gist.github.com/berleon/614468b66327b6ac9396785c60640dd8.
    """

    action_space = {
    # | State      | Action           | Reward  | Next state probs | 
    # |------------|------------------|---------|------------------| 
       "Class 1":  {
                    "Study":          ( -2.,    {
                                                 "Class 2": 1.
                                                 }),
                    "Go on Facebook": ( -1.,    {
                                                 "Facebook": 1.
                                                 })
                    },
       "Class 2":  {
                    "Study":          ( -2.,    {
                                                 "Class 3": 1.
                                                 }),
                    "Fall asleep":    ( 0.,     {
                                                 "Asleep": 1.
                                                 })
                    },
       "Class 3":  {
                    "Study":          ( 10.,    {
                                                 "Pass": 1.
                                                 }),
                    "Go to the pub":  ( 1.,     {
                                                 "Pub": 1.
                                                 })
                    },
       "Facebook": {
                    "Keep scrolling": ( -1.,    {
                                                 "Facebook": 1.
                                                 }),
                    "Close Facebook": ( -2.,    {
                                                 "Class 1": 1.
                                                 }) 
                    },
       "Pub":      {
                    "Have a pint":    ( -2.,    {
                                                 "Class 1": 0.2,
                                                 "Class 2": 0.4,
                                                 "Class 3": 0.4
                                                 })
                    },
       "Pass":     {
                    "Fall asleep":    ( 0.,     {
                                                 "Asleep": 1.
                                                 })
                   },
       "Asleep":   {
                    "Stay asleep":    ( 0.,     {
                                                 "Asleep": 1.
                                                 })
                    }
    }

    def __init__(self, verbose):
        self.verbose = verbose
        self.ep = 0

    def reset(self):
        self.ep += 1
        self.t = 0
        self.state = "Class 1"
        if self.verbose:
            print("="*27+f" EPISODE {str(self.ep).rjust(3)} "+"="*27)
            self.print_header()
        return self.state

    def step(self, action):
        reward = self.reward(self.state, action)
        next_state = self.next_state(self.state, action)
        done = (next_state == "Asleep")
        if self.verbose:                
            print((f"| {str(self.t).ljust(5)} | {self.state.ljust(8)} | {action.ljust(14)} |"
                   f"{str(reward).rjust(5).ljust(7)} | {next_state.ljust(10)} | {str(done).ljust(5)} |"))
        self.state = next_state
        self.t += 1
        return self.state, reward, done, {}

    def reward(self, state, action):
        self.check_valid(state, action)
        return self.action_space[state][action][0]

    def next_state(self, state, action):
        self.check_valid(state, action)
        nextstate_options = self.action_space[state][action][1]
        return choice(list(nextstate_options.keys()), p=list(nextstate_options.values()))

    def check_valid(self, state, action):
        assert action in self.action_space[state], f"Invalid action '{action}' in state '{state}'"

    def print_header(self):
        print("| Time  | State    | Action         | Reward | Next state | Done  |")
        print("|-------|----------|----------------|--------|------------|-------|")