import numpy as np
import torch as T
from dqn_components import DeepQNetwork
from dqn_components import ReplayBuffer

class Agent:
    '''
    Deep QN agent class
    '''
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size,
                 batch_size, eps_min=0.01, eps_dec=5e-7, replace=1000,
                 env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(lr=self.lr, n_actions=self.n_actions,
                                   name=self.env_name+'_q_eval',
                                   input_dims=self.input_dims,
                                   chkpt_dir=self.chkpt_dir
                                   )

    def choose_action(self, observation):
        '''
        action selection mechanism
        '''
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        '''
        stores data in agent memory
        '''
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        '''
        samples data from memory for learning process
        '''
        state, action, reward, state_, done = \
                                        self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(state_).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def save_models(self):
        self.q_next.save_checkpoint()
        self.q_eval.save_checkpoint()

    def load_models(self):
        self.q_next.load_checkpoint()
        self.q_eval.load_checkpoint()

    def learn(self):
        '''
        updates parameters of evaluation q network
        '''
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_eval(states_)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min





