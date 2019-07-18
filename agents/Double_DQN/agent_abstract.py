import torch
import numpy as np
import random

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Agent_segment(object):
    def __init__(self, epsilon, segment_length, random_can_stop=False):
        self.epsilon = epsilon
        self.segment_length = segment_length
        self.random_can_stop = random_can_stop

    def set_epsilon(self, e):
        self.epsilon = e

    def act(self, s, train=True, must_stop=False):
        """ This function should return the next action to do:
        an array of length self.segment_length with a random exploration of epsilon"""
        if train and np.random.rand() <= self.epsilon:
            if must_stop:
                return torch.tensor([1, random.randrange(self.segment_length)], device=device, dtype=torch.long)
            if self.random_can_stop:
                return torch.tensor([random.randrange(2), random.randrange(self.segment_length)], device=device,
                                    dtype=torch.long)
            return torch.tensor([0, random.randrange(self.segment_length)], device=device, dtype=torch.long)
        a = self.learned_act(s)
        if must_stop:
            return torch.tensor([1, torch.argmax(a[:, 1, :])], device=device, dtype=torch.long)
        max_index = torch.argmax(a)
        return torch.tensor([max_index / self.segment_length, max_index % self.segment_length], device=device,
                            dtype=torch.long)

    def learned_act(self, s):
        """ Act via the policy of the agent, from a given state s
        it proposes an action a"""
        pass

    def reinforce(self, s, n_s, a, r, game_over_):
        """ This function is the core of the learning algorithm.
        It takes as an input the current state s_, the next state n_s_
        the action a_ used to move from s_ to n_s_ and the reward r_.

        Its goal is to learn a policy.
        """
        pass

    def save(self):
        """ This function returns basic stats if applicable: the
        loss and/or the model"""
        pass

    def load(self):
        """ This function allows to restore a model"""
        pass


class Agent_metal(object):
    def __init__(self, epsilon, action_length):
        self.epsilon = epsilon
        self.action_length = action_length

    def set_epsilon(self, e):
        self.epsilon = e

    def act(self, s, train=True):
        """ This function should return the next action to do:
        an array of length self.action_length with a random exploration of epsilon"""
        if train and np.random.rand() <= self.epsilon:
            return torch.tensor(random.randrange(self.action_length), device=device, dtype=torch.long)

        a = self.learned_act(s)
        max_index = torch.argmax(a)
        return torch.tensor(max_index, device=device, dtype=torch.long)

    def learned_act(self, s):
        """ Act via the policy of the agent, from a given state s
        it proposes an action a"""
        pass

    def reinforce(self, s, n_s, a, r, game_over_):
        """ This function is the core of the learning algorithm.
        It takes as an input the current state s_, the next state n_s_
        the action a_ used to move from s_ to n_s_ and the reward r_.

        Its goal is to learn a policy.
        """
        pass

    def save(self):
        """ This function returns basic stats if applicable: the
        loss and/or the model"""
        pass

    def load(self):
        """ This function allows to restore a model"""
        pass