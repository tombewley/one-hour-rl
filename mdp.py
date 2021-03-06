from numpy.random import choice


def sample_from_dict(dict_):
    return choice(list(dict_.keys()), p=list(dict_.values()))

    
class TabularMDP:
    """
    Generic OpenAI Gym-compatible class for tabular MDPs.
    Requires self._spec to be set to a dictionary with the following structure:

    {[str: state]: {
        "p_init": [float: marginal probability of initialising in 'state'],
        "actions": {
            [str: action]: (
                [float: reward for taking 'action' in 'state',
                {
                    [str: next state]: [float: probability of transitioning to 'next state' after taking 'action' in 'state'],
                    ... more next states ...
                    }
                ),
            ... more actions ...
            }
        },
    ... more states ...
    }

    Note that a state is taken as terminal if its action space is empty.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.ep = 0

    def reset(self):
        self.ep += 1
        self.t = 0
        self.state = self.sample_initial_state()
        if self.verbose:
            print("="*27+f" EPISODE {str(self.ep).rjust(3)} "+"="*27)
            self.print_header()
        return self.state

    def step(self, action):
        reward = self.reward(self.state, action)
        next_state = self.sample_next_state(self.state, action)
        done = self.is_terminal(next_state)
        if self.verbose:                
            print((f"| {str(self.t).ljust(5)} | {self.state.ljust(8)} | {action.ljust(14)} |"
                   f"{str(reward).rjust(5).ljust(7)} | {next_state.ljust(10)} | {str(done).ljust(5)} |"))
        self.state = next_state
        self.t += 1
        return self.state, reward, done, {}

    def initial_probs(self):
        return {state: v["p_init"] for state, v in self._spec.items()} 

    def sample_initial_state(self):
        return sample_from_dict(self.initial_probs())

    def action_space(self, state):
        return set(self._spec[state]["actions"].keys())

    def reward(self, state, action):
        return self._spec[state]["actions"][action][0]

    def transition_probs(self, state, action):
        return self._spec[state]["actions"][action][1]

    def sample_next_state(self, state, action):
        return sample_from_dict(self.transition_probs(state, action))

    def is_terminal(self, state):
        return not self._spec[state]["actions"] 

    def terminal_states(self):
        return {state for state in self._spec if self.is_terminal(state)} 

    def print_header(self):
        print("| Time  | State    | Action         | Reward | Next state | Done  |")
        print("|-------|----------|----------------|--------|------------|-------|")


class StudentMDP(TabularMDP):
    """
    OpenAI Gym-compatible class for the Student MDP from David Silver's "Introduction to Reinforcement Learning" class.
    See images/student-mdp.png for a diagram of this MDP.
    Specification adapted from https://gist.github.com/berleon/614468b66327b6ac9396785c60640dd8.
    """

    _spec = {
    # | State      | Action           | Reward  | Next state probs | 
    # |------------|------------------|---------|------------------| 
       "Class 1": {
        "p_init": 1.,
        "actions": {
                    "Study":          ( -2.,    {
                                                 "Class 2": 1.
                                                 }),
                    "Go on Facebook": ( -1.,    {
                                                 "Facebook": 1.
                                                 })
                    }},
       "Class 2": {
        "p_init": 0.,
        "actions": {
                    "Study":          ( -2.,    {
                                                 "Class 3": 1.
                                                 }),
                    "Fall asleep":    ( 0.,     {
                                                 "Asleep": 1.
                                                 })
                    }},
       "Class 3": {
        "p_init": 0.,
        "actions": {
                    "Study":          ( 10.,    {
                                                 "Pass": 1.
                                                 }),
                    "Go to the pub":  ( 1.,     {
                                                 "Pub": 1.
                                                 })
                    }},
       "Facebook": {
        "p_init": 0.,
        "actions": {
                    "Keep scrolling": ( -1.,    {
                                                 "Facebook": 1.
                                                 }),
                    "Close Facebook": ( -2.,    {
                                                 "Class 1": 1.
                                                 }) 
                    }},
       "Pub": {
        "p_init": 0.,
        "actions": {
                    "Have a pint":    ( -2.,    {
                                                 "Class 1": 0.2,
                                                 "Class 2": 0.4,
                                                 "Class 3": 0.4
                                                 })
                    }},
       "Pass": {
        "p_init": 0.,
        "actions": {
                    "Fall asleep":    ( 0.,     {
                                                 "Asleep": 1.
                                                 })
                   }},
       "Asleep": {
        "p_init": 0.,
        "actions": {}
                    }
    }