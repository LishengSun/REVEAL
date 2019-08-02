import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import json
import pdb
import copy
import optparse
import math
from itertools import count
import sys
import pandas as pd


import random
import matplotlib.pyplot as plt

from distributions import Categorical, DiagGaussian
from collections import namedtuple

from context import MNIST_env
from MNIST_env import img_env_brightest_patches
import utils
from visualization_utils import *
import seaborn as sns
import imageio
import argparse, os


parser = argparse.ArgumentParser("Learn policy using Double DQN")
parser.add_argument('--env', type=str, default='MNIST_brightest', help=['CartPole-v0'])
parser.add_argument('--num-episodes', default=2000, type=int)
parser.add_argument('--save-interval', default=200, type=int)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--lr', default=10e-5, type=float)
parser.add_argument('--eps-start', default=0.15, type=float, help='epsilon greedy')
parser.add_argument('--eps-end', default=0.05, type=float, help='epsilon greedy')
parser.add_argument('--eps-decay', default=200, type=int, help='epsilon greedy')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--reward-shapping', default='simple', type=str, help='which reward shapping to use in the env.')
# parser.add_argument('--mirror', action='store_true')
parser.add_argument('--out-dir-fig', default='./double_dqn/figs')
parser.add_argument('--out-dir-models', default='./double_dqn/models')
parser.add_argument('--out-dir-files', default='./double_dqn/files')

parser.add_argument('--replay-memory', type=int, default=10000)
parser.add_argument('--num-runs', type=int, default=3)
parser.add_argument('--num-steps', type=int, default=49)
parser.add_argument('--num-labels', type=int, default=2)
parser.add_argument('--num-targets', type=int, default=6)
parser.add_argument('--window-size', type=int, default=5)
parser.add_argument('-f','--file', help='Path for input file. This is a trick for jupyter notebook')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def smoothing_average(x, factor=10):
# 	running_x = 0
# 	for i in range(len(x)):
# 		U = 1. / min(i+1, factor)
# 		running_x = running_x * (1 - U) + x[i] * U
# 		x[i] = running_x
# 	return x



class DQN(nn.Module):

	def __init__(self, input_dim, action_space, conv):
		super(DQN, self).__init__()
		# self.conv = conv
		# if self.conv:
		self.conv1 = nn.Conv2d(input_dim, 10, kernel_size=5, stride=2)
			# self.bn1 = nn.BatchNorm2d(10)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
			# self.bn2 = nn.BatchNorm2d(20)
		self.conv3 = nn.Conv2d(20, 10, kernel_size=5)
			# self.bn3 = nn.BatchNorm2d(10)
		# else:
		self.linear = nn.Linear(360, 512)
		self.head = nn.Linear(512, action_space)

	def forward(self, x):
		
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.linear(x.view(x.size(0),-1)))
		return self.head(x.view(x.size(0), -1))


def Qs_to_value_image(Qs, window_size):
	"""
		from list of Q values to 32*32 image
	"""
	value_image = np.zeros((32, 32))
	num_row_patches = int(32 / window_size) + 1
	num_col_patches = int(32 / window_size) + 1
	i = 0
	for rp in range(num_row_patches):
		for cp in range(num_col_patches):
			value_image[window_size*rp:window_size*(rp+1), window_size*cp:window_size*(cp+1)] = Qs[i]
			i += 1
	return value_image


def select_action(env, i_episode, policy_net, train=True, allow_repeat_action=True):
	"""
		during training: allow the agent to choose same action more than once (allow_repeat_action=True)
						keep greedy epsilon with epsilon decaying with episodes
		during testing: epsilon = 0, can choose allow_repeat_action=True / False
	"""
	
	sample = random.random()
	
	if train: # keep eps greedy in train time
		eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
			math.exp(-1. * i_episode / args.eps_decay)
		allow_repeat_action = True
	else: # remove eps greedy in test time
		eps_threshold = 0

	if sample > eps_threshold:
		with torch.no_grad():
			Qs = policy_net(torch.Tensor([env.state]).to(device))
			if allow_repeat_action:
				return Qs[0], Qs.max(1)[1].view(1, 1), Qs.max(1)[0].item() # Q values, action, max Q values
			else:
				action_candidates = Qs.argsort(descending=True)
				for cand in action_candidates[0]:
					if not cand.item() in env.action_history:
						return Qs[0], cand.view(1,1), Qs[0][cand.item()].item() # Q values, action, selected Q values
	else: # random move
		return np.zeros((env.action_space.n)), torch.tensor([[random.randrange(49)]], device=device, dtype=torch.long), np.nan#, policy_net(state).max(1)[0].item() # max_Q is nan for random selection
	
class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))



def optimize_model():
	coin_flip = np.random.random()
	policy_net_tmp = policy_net
	target_net_tmp = target_net
	if coin_flip > 0.5:
		# print ('optimizing target_net')
		policy_net_tmp = target_net
		target_net_tmp = policy_net
	# else:
		# print ('optimizing policy_net')
	for p in policy_net_tmp.parameters():
		p.requires_grad = True
	for p in target_net_tmp.parameters():
		p.requires_grad = False

	if len(memory) < args.batch_size:
		# print ('no enough examples for optimizing')
		loss = 9999
		return policy_net, target_net
	transitions = memory.sample(args.batch_size)
	batch = Transition(*zip(*transitions))

	# Compute a mask of non-final states and concatenate the batch elements
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), dtype=torch.uint8).to(device)

	non_final_next_states = torch.stack([s for s in batch.next_state \
		if s is not None]).to(device)
	# print ('non_final_next_states', non_final_next_states.shape)
	state_batch = torch.stack(batch.state).to(device)
	
	action_batch = torch.LongTensor(torch.stack(batch.action)).to(device)
	reward_batch = torch.cat(batch.reward).to(device)
	
		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken
	# pdb.set_trace()
	state_action_values = policy_net_tmp(
		state_batch.float()).gather(1, action_batch)#.unsqueeze(-1))

		# Compute V(s_{t+1}) for all next states.
	next_state_values = torch.zeros(args.batch_size, device=device)
	with torch.no_grad():
		next_actions_batch = policy_net_tmp(
			non_final_next_states.float()).max(1)[1].unsqueeze(1)
	next_state_values[non_final_mask] = target_net_tmp(
		non_final_next_states.float()).gather(
		1, next_actions_batch).squeeze().detach()
		# Compute the expected Q values
	expected_state_action_values = (
		next_state_values * args.gamma) + reward_batch

		# Compute Huber loss
	td_error = 0.5 * F.mse_loss(
		state_action_values, expected_state_action_values.unsqueeze(1),
		reduction='none').squeeze()
		# if args.prioritized_replay:
		# 	q_loss = torch.Tensor(batch['weights']).to(device) * td_error
		# 	new_priorities = q_loss.cpu().data.numpy() + 1e-6
		# 	memory.update_priorities(batch['inds'], new_priorities)
		# 	loss = q_loss.mean()
		# else:
	loss = td_error.mean()

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	for param in policy_net_tmp.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

	# print ('loss from optimizing: ', loss)
	return policy_net_tmp, target_net_tmp




if __name__ == '__main__':
	env = img_env_brightest_patches.ImgEnv('mnist', train=True, max_steps=args.num_steps, \
		channels=2, window=args.window_size, num_labels=args.num_labels, \
		reward_shapping=args.reward_shapping, num_targets=args.num_targets)

	input_dim = env.observation_space.shape[0]
	action_space = env.action_space.n
	
	args_param = vars(args)
	toprint = ['lr', 'num_episodes', 'window_size', 'num_labels', 'reward_shapping', 'num_targets'] #, 'default']
	toprint.sort()
	save_name = args.env + '_'
	for arg in toprint:
	    if args_param[arg] not in [True, False, None] or args_param[arg]==1:
	        save_name += '{}{}_'.format(arg, args_param[arg])
	    elif args_param[arg] is True:
	        save_name += '{}_'.format(arg)


	run_episode_durations = []
	run_maxQs = []
	run_rewards = []

	for run in range(args.num_runs):
		policy_net = DQN(input_dim, action_space, conv=True).to(device)
		target_net = DQN(input_dim, action_space, conv=True).to(device)

		optimizer = optim.RMSprop(
			list(policy_net.parameters()) + list(target_net.parameters()),
			lr=args.lr)

		memory = ReplayMemory(10000)
		episode_durations = []
		episode_maxQs = []
		episode_reward = []
		for i_episode in range(args.num_episodes):

			if (i_episode + 1) % 10 == 0:
				print("\rEpisode {}/{} Duration {} Reward {}.\n".format(
					i_episode + 1, args.num_episodes,
					env.num_steps, ep_rew), end="")
			
			if (i_episode + 1) % args.save_interval == 0:
				print("\rEpisode {}/{} Reward {} .\n".format(
					i_episode + 1, args.num_episodes,
					ep_rew), end="")
				sys.stdout.flush()
				torch.save(policy_net.state_dict(), os.path.join(args.out_dir_models, 'model'+save_name+'e%i_run%i.pth'%(i_episode, run)))
				torch.save(optimizer.state_dict(), os.path.join(args.out_dir_models, 'optimizer'+save_name+'e%i_run%i.pth'%(i_episode, run)))

			# Initialize the environment and state
			ep_rew = 0
			state = env.reset()
			max_Qs = []
			for t in range(args.num_steps):
				# Select and perform an action
				Qs, action, max_Q = select_action(env, i_episode, policy_net)
				next_state, reward, done, _ = env.step(action.item())
				
				max_Qs.append(max_Q)
				ep_rew += args.gamma * reward
				policy_net, target_net = optimize_model()
				
				# axarr1 = plt.subplot(211)
				# axarr1.imshow(state[0, :, :], extent=[0, 32, 32, 0], vmin=0, vmax=1) #alpha=0.6, 
				# axarr2 = plt.subplot(212)
				# axarr2.imshow(next_state[0, :, :], extent=[0, 32, 32, 0], vmin=0, vmax=1) #alpha=0.6, 
				# plt.show()

				if not done:
					# print ('next_state = 1')
					memory.push(torch.from_numpy(state), torch.from_numpy(action.cpu().numpy()[0]), \
					torch.from_numpy(next_state), torch.tensor([reward]).float())
				else:
					next_state = None
					# print ('next_state', next_state)
					memory.push(torch.from_numpy(state), torch.from_numpy(action.cpu().numpy()[0]), \
					next_state, torch.tensor([reward]).float())
					break
				
				state = next_state
				
			episode_durations.append(env.num_steps)
			episode_reward.append(ep_rew)
			episode_maxQs.append(max(max_Qs))
			
		run_episode_durations.append(episode_durations)	
		run_maxQs.append(episode_maxQs)
		run_rewards.append(episode_reward)

	plt.subplot(311)
	plt.xlabel('Episode')
	plt.ylabel('Episode_Duration')
	sns.tsplot(data=[smoothing_average(run_episode_durations[i], \
		100) for i in range(len(run_episode_durations))], \
		time=list(range(args.num_episodes)), ci=[68, 95], \
		ax=plt.subplot(3, 1, 1), color='red', condition='train')
	plt.ylim([0, 20])
	plt.title(save_name[:-1])


	plt.subplot(312)
	plt.xlabel('Episode')
	plt.ylabel('max Q')
	sns.tsplot(data=[run_maxQs[i] for i in range(len(run_maxQs))], \
		time=list(range(args.num_episodes)), ci=[68, 95], \
		ax=plt.subplot(3, 1, 2), color='red', condition='train')

	plt.subplot(313)
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	sns.tsplot(data= [smoothing_average(run_rewards[i], 100) for i in range(len(run_rewards))], \
		time=list(range(args.num_episodes)), ci=[68, 95], \
		ax=plt.subplot(3, 1, 3), color='red', condition='train')

	ts = int(time.time())
	fig_name = save_name+'doubledqn_plot%s.png' % ts
	plt.savefig(os.path.join(args.out_dir_fig, fig_name))
	plt.close()

	ind = ['run_%i'%i for i in range(args.num_runs)]
	col = ['epi_%i'%i for i in range(args.num_episodes)]
	run_episode_durations_df = pd.DataFrame(data=np.array(run_episode_durations), index=ind, columns=col)
	run_maxQs_df = pd.DataFrame(data=np.array(run_maxQs), index=ind, columns=col)
	run_rewards_df = pd.DataFrame(data=np.array(run_rewards), index=ind, columns=col)

	run_episode_durations_df.to_csv(os.path.join(args.out_dir_files, save_name[:-1]+'_run_episode_durations'), index=ind)
	run_maxQs_df.to_csv(os.path.join(args.out_dir_files, save_name[:-1]+'_run_maxQs'), index=ind)
	run_rewards_df.to_csv(os.path.join(args.out_dir_files, save_name[:-1]+'_run_rewards'), index=ind)


	# run_episode_durations_df = pd.read_csv(os.path.join(args.out_dir_files, save_name[:-1]+'_run_episode_durations'), index_col=0)

	# run_episode_durations = run_episode_durations_df.values
