import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import json
import pdb
import copy
import optparse

import random
import matplotlib.pyplot as plt

from distributions import Categorical, DiagGaussian
from collections import namedtuple

from context import MNIST_env
from MNIST_env import img_env_rank

import utils
import seaborn as sns
import pdb


import model

from PIL import Image

from random import randint
import numpy as np
import os, time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def smoothing_average(x, factor=100):
	running_x = 0
	X = copy.deepcopy(x)
	for i in range(len(X)):
		U = 1. / min(i+1, factor)
		running_x = running_x * (1 - U) + X[i] * U
		X[i] = running_x
	return X




class myNet(nn.Module):
	def __init__(self, obs_shape, action_space, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
		super(myNet, self).__init__()
		self.dataset = dataset
		if len(obs_shape) == 3: #our mnist case
			self.base = model.CNNBase(obs_shape[0], recurrent_policy, dataset=dataset)
		elif len(obs_shape) == 1:
			assert not recurrent_policy, \
				"Recurrent policy is not implemented for the MLP controller"
			self.base = MLPBase(obs_shape[0])
		else:
			raise NotImplementedError

		if action_space.__class__.__name__ == "Discrete": # our case
			num_outputs = action_space.n
			self.dist = Categorical(self.base.output_size+action_space.n, num_outputs)
		elif action_space.__class__.__name__ == "Box":
			num_outputs = action_space.shape[0]
			self.dist = DiagGaussian(self.base.output_size, num_outputs)
		else:
			raise NotImplementedError

		if dataset in ['mnist', 'cifar10']:
			self.clf = Categorical(self.base.output_size, action_space.n) 

		self.state_size = self.base.state_size

	def forward(self, inputs, states, masks):
		raise NotImplementedError

	def act(self, inputs, states, masks, deterministic=False):
		actor_features, states = self.base(inputs, states, masks)
		self.actor_features = actor_features

		if self.dataset in img_env_rank.IMG_ENVS:
			clf = self.clf(self.actor_features)
			clf_proba = clf.probs

			# classif = clf_proba.sort()[-1][0][-3:] # -1=idx, 0=first example -3:=3 highest proba

			if deterministic:
				classif = clf.mode()
			else:
				classif = clf.sample()
			

		actor_features_with_clf_proba = torch.cat((actor_features, clf_proba), 1) 
		dist = self.dist(actor_features_with_clf_proba)
		Q_values = dist.logits
		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		action_log_probs = dist.log_probs(action)

		action = torch.cat([action, classif], 1)

		return action, Q_values, clf_proba, states #dist.logits = Q values



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
						('state', 'action', 'next_state', 'reward', 'target'))



def optimize_myNet(net, optimizer, BATCH_SIZE=128, OPTIMIZE_CLF=False):
	if len(memory) < BATCH_SIZE:
		return
	print ('Optimizing')
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))


	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), dtype=torch.uint8).to(device)

	non_final_next_states = torch.stack([s for s in batch.next_state \
		if s is not None]).to(device)
	# print ('non_final_next_states', non_final_next_states.shape)
	state_batch = torch.stack(batch.state).to(device)
	
	action_batch = torch.stack(batch.action).to(device)
	reward_batch = torch.cat(batch.reward).to(device)

	
	_, Q_values_batch, clf_proba_batch, _ = net.act(
		inputs=state_batch.float(),
		states=state_batch, masks=state_batch[1])
	

	
	state_action_values = Q_values_batch.gather(1, action_batch[:,0].view(BATCH_SIZE,1))

	next_state_values = torch.zeros(BATCH_SIZE).to(device)

	_, next_Q_values_batch, _, _= target_net.act(\
		inputs=non_final_next_states.float(),\
		states=non_final_next_states, masks=non_final_next_states[1])
	
	
	next_state_values[non_final_mask] = next_Q_values_batch.max(1)[0].detach()
	
	expected_state_action_values = next_state_values * GAMMA + reward_batch # Compute the expected Q values


	loss_navig = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
	
	# target_prob_batch = torch.stack(batch.target_prob).to(device)
	target_batch = torch.cat(batch.target).to(device)
	
	loss_clf = F.nll_loss(torch.log(clf_proba_batch), target_batch)
	# pdb.set_trace()
	if OPTIMIZE_CLF:
		total_loss = 0.1*loss_clf + 0.9*loss_navig
	else:
		total_loss = loss_navig  #only train on navigation loss
	# total_loss = 0.1*loss_clf + 0.9*loss_navig

	optimizer.zero_grad()
	total_loss.backward()
	# pdb.set_trace()
	# for name, param in net.named_parameters():
		# print(name, torch.max(torch.abs(param.grad)).data if param.grad is not None else None)
	for param in filter(lambda p: p.requires_grad, net.parameters()):
		param.grad.data.clamp_(-1, 1) #gradient clipping prevent only the exploding gradient problem

	
	optimizer.step()

	# print ('total_loss, loss_clf, loss_navig', total_loss.detach().item(), loss_clf.detach().item(), loss_navig.detach().item())
	return total_loss, loss_clf, loss_navig



####################### TRAINING ############################
if __name__ == '__main__':

	# Collect arguments 
	parser = optparse.OptionParser()
	parser.add_option('--OPTIMIZE_CLF', action="store_true", \
		dest="OPTIMIZE_CLF", help='whether to also optimize the classification loss')

	
	parser.set_defaults(OPTIMIZE_CLF=False)

	options, args = parser.parse_args()

	OPTIMIZE_CLF = options.OPTIMIZE_CLF



	BATCH_SIZE = 128
	NUM_STEPS = 49
	GAMMA = 1 - (1 / NUM_STEPS) # Set to horizon of max episode length
	EPS = 0.05
	NUM_LABELS = 2
	WINDOW_SIZE = 5
	NUM_EPISODES = 10000
	TARGET_UPDATE = 10
	RUNS = 1
	MODEL_DIR = './trained_model/'
	RESULT_DIR = './results/'

	env = img_env_rank.ImgEnv('mnist', train=True, max_steps=NUM_STEPS, channels=2, window=WINDOW_SIZE, num_labels=NUM_LABELS)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	run_durations = []
	run_total_rewards = []
	run_loss_clf = []
	for run in range(RUNS):
		net = myNet(\
			obs_shape=env.observation_space.shape, \
			action_space=env.action_space, dataset='mnist').to(device)
		target_net = myNet(\
			obs_shape=env.observation_space.shape, \
			action_space=env.action_space, dataset='mnist').to(device)
		target_net.load_state_dict(net.state_dict())
		target_net.eval()
		memory = ReplayMemory(10000)

		total_rewards = []
		episode_durations = []
		loss_classification = []
		q_values = []
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, amsgrad=True)

		for i_episode in range(NUM_EPISODES):
			print ('run %i, episode %i'%(run, i_episode))
			total_reward_i = 0
			observation = env.reset()
			target = env.target

			for t in range(NUM_STEPS): # allow 100 steps
				actionS, Q_values, clf_proba, states = net.act(inputs=torch.from_numpy(observation).float().resize_(1, 2, 32, 32).to(device), \
					states=observation, masks=observation[1])
				actionS = actionS.cpu().numpy()[0]
				patch_pred = actionS[1]
				last_observation = observation
				rand = np.random.rand()
				if rand < EPS:
					actionS = np.array(
						[np.random.choice(range(env.action_space.n)), np.random.choice(range(env.action_space.n))])
				action = actionS[0]
				observation, reward, done, info = env.step(actionS)
				total_reward_i = reward + GAMMA*total_reward_i
				memory.push(torch.from_numpy(last_observation), torch.from_numpy(actionS), \
					torch.from_numpy(observation), torch.tensor([reward]).float(), torch.tensor([target]))
				optimize_myNet(net, optimizer, BATCH_SIZE, OPTIMIZE_CLF)

				if done:
				# print ('Done after %i steps'%(t+1))
					break
				# Update the target network, copying all weights and biases in DQN
			if i_episode % TARGET_UPDATE == 0:
				target_net.load_state_dict(net.state_dict())

			loss_classification_i = F.nll_loss(torch.log(clf_proba), torch.tensor([env.target]).to(device))
			total_rewards.append(total_reward_i)
			episode_durations.append(t+1)
			loss_classification.append(loss_classification_i.detach().item())
			q_values.append(Q_values)
		run_durations.append(episode_durations)
		run_total_rewards.append(total_rewards)
		run_loss_clf.append(loss_classification)

	torch.save(net.state_dict(), os.path.join(MODEL_DIR,'model_img_env_rank_OPT_CLF{OPTIMIZE_CLF}_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws.pth'.format(**locals())))
	torch.save(optimizer.state_dict(), os.path.join(MODEL_DIR,'optimizer_img_env_rank_OPT_CLF{OPTIMIZE_CLF}_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws.pth'.format(**locals())))


	plt.figure(1)
	plt.title('only train on loss_dist')
		
	plt.subplot(411)
	plt.xlabel('Episode')
	plt.ylabel('Episode_Duration')
	durations_t = torch.tensor(episode_durations[0], dtype=torch.float)
	plt.plot(smoothing_average(run_durations[0]))

	plt.subplot(412)
	plt.xlabel('Episode')
	plt.ylabel('Rewards')
	total_rewards_t = torch.tensor(total_rewards[0], dtype=torch.float)
	plt.plot(smoothing_average(run_total_rewards[0]))

	plt.subplot(413)
	plt.xlabel('Episode')
	plt.ylabel('Loss Classification')
	loss_classification_t = torch.tensor(loss_classification[0], dtype=torch.float)
	plt.plot(smoothing_average(run_loss_clf[0]))

	plt.subplot(414)
	plt.xlabel('Episode')
	plt.ylabel('max Q')
	q_value_max = [np.max(q_values[i].cpu().detach().numpy(), axis=1) for i in range(NUM_EPISODES)]

	plt.plot(smoothing_average(q_value_max))
	plt.savefig(os.path.join(RESULT_DIR,'DQN_img_env_rank_OPT_CLF{OPTIMIZE_CLF}_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws'.format(**locals())))
	plt.show()
		# sns.tsplot(data=smoothing_average(q_value_max), time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 2), color='red')

	# plt.subplot(514)
	# plt.xlabel('Episode')
	# plt.ylabel('Nvg loss')
	# plt.plot(smoothing_average(loss_navig[50:], factor=100))


	# plt.subplot(515)
	# plt.xlabel('Episode')
	# plt.ylabel('Clf loss')
	# plt.plot(smoothing_average(loss_clf[50:], factor=100))


