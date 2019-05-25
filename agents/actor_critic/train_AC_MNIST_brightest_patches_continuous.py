import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from distributions import Categorical, DiagGaussian

from context import MNIST_env
from MNIST_env import img_env_brightest_patches_continuous
import model


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
					help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
args = parser.parse_args()




SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
	def __init__(self, obs_shape, action_space, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
		super(Policy, self).__init__()
		self.dataset = dataset
		if len(obs_shape) == 3: #our mnist case
			self.base = model.CNNBase(obs_shape[0], recurrent_policy, dataset=dataset)
		else:
			raise NotImplementedError

		if action_space.__class__.__name__ == "Box":
			num_outputs = 1 # one value for row or col, then have 2 action heads
			self.row_head = DiagGaussian(self.base.output_size, num_outputs)
			self.col_head = DiagGaussian(self.base.output_size, num_outputs)
			self.value_head = nn.Linear(self.base.output_size, 1)

		else:
			raise NotImplementedError



	def forward(self, x):
		raise NotImplementedError

	def act(self, x):
		x = self.base(x)
		row_score, row_prob = self.row_head(x)
		col_score, col_prob = self.col_head(x)
		state_values = self.value_head(x)

		return row_prob, col_prob, state_values
	  





if __name__ == '__main__':
	MAX_STEPS = 16
	WINDOW = 8
	env = img_env_brightest_patches_continuous.ImgEnv('mnist', train=True, max_steps=MAX_STEPS, channels=2, window=WINDOW, num_labels=10)

	policy = Policy(env.observation_space.shape, env.action_space)
	optimizer = optim.Adam(policy.parameters(), lr=3e-2)
	eps = np.finfo(np.float32).eps.item()


























