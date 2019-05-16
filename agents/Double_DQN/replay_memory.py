# Memory Replay
from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'game_over'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def remember(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size <= self.__len__():
            return random.sample(self.memory, batch_size)
        return random.choices(self.memory, k=batch_size) 