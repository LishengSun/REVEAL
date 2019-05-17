import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import matplotlib.pyplot as plt
import seaborn as sns

from distributions import Categorical, DiagGaussian
from collections import namedtuple, defaultdict

from context import MNIST_env
from MNIST_env import img_env_orig

import utils

import model

from PIL import Image

from random import randint
import numpy as np
import os




def smoothing_average(x, factor=10):
	running_x = 0
	for i in range(len(x)):
		U = 1. / min(i+1, factor)
		running_x = running_x * (1 - U) + x[i] * U
		x[i] = running_x
	return x




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
			self.dist = Categorical(self.base.output_size, num_outputs)
		elif action_space.__class__.__name__ == "Box":
			num_outputs = action_space.shape[0]
			self.dist = DiagGaussian(self.base.output_size, num_outputs)
		else:
			raise NotImplementedError

		if dataset in ['mnist', 'cifar10']:
			self.clf = Categorical(self.base.output_size, 2)#10)

		self.state_size = self.base.state_size

	def forward(self, inputs, states, masks):
		raise NotImplementedError

	def act(self, inputs, states, masks, deterministic=False):
		actor_features, states = self.base(inputs, states, masks)
		self.actor_features = actor_features
		dist = self.dist(actor_features)
		Q_values = dist.logits
		if deterministic:
			action = dist.mode()
		else:
			action = dist.sample()

		action_log_probs = dist.log_probs(action)
		if self.dataset in img_env_orig.IMG_ENVS:
			clf = self.clf(self.actor_features)
			clf_proba = clf.logits
			if deterministic:
				classif = clf.mode()
			else:
				classif = clf.sample()
			action = torch.cat([action, classif], 1)
			action_log_probs += clf.log_probs(classif)

		return action, Q_values, clf_proba, action_log_probs, states #dist.logits = Q values


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
						('state', 'action', 'next_state', 'reward', 'curr_label'))






def optimize_myNet(net, curr_label, optimizer, BATCH_SIZE=128, optimize_clf=False):
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))


	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
										  batch.next_state)), dtype=torch.uint8).to(device)

	non_final_next_states = torch.stack([s for s in batch.next_state \
		if s is not None]).to(device)
	# print ('non_final_next_states', non_final_next_states.shape)
	state_batch = torch.stack(batch.state).to(device)
	# print ('state_batch.size', state_batch.size)
	action_batch = torch.stack(batch.action).to(device)
	reward_batch = torch.cat(batch.reward).to(device)


	_, Q_values_batch, clf_proba_batch, _, _ = net.act(
		inputs=state_batch.float(),
		states=state_batch, masks=state_batch[1])

	state_action_values = Q_values_batch.gather(1, action_batch[:, 0].view(BATCH_SIZE,1))
	next_state_values = torch.zeros(BATCH_SIZE).to(device)

	_, next_Q_values_batch, _, _, _= target_net.act(inputs=non_final_next_states.float(),states=non_final_next_states, masks=non_final_next_states[1])

	next_state_values[non_final_mask] = next_Q_values_batch.max(1)[0].detach()

	expected_state_action_values = (next_state_values * GAMMA) + reward_batch # Compute the expected Q values
	loss_dist = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	curr_label_batch = torch.cat(batch.curr_label).to(device)
	loss_clf = F.nll_loss(clf_proba_batch, curr_label_batch)

	total_loss = loss_dist + loss_clf
# 	optimizer_dist = optim.RMSprop(net.parameters())
# 	optimizer_dist.zero_grad()
# 	total_loss.backward()
# 	for param in net.dist.parameters():
# 		param.grad.data.clamp_(-1, 1)
# 	optimizer_dist.step()
	optimizer.zero_grad()
	total_loss.backward()
	for param in net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

	return total_loss, loss_clf, loss_dist




####################### TRAINING ############################
if __name__ == '__main__':
	BATCH_SIZE = 128
	NUM_STEPS = 10
	GAMMA = 1 - (1 / NUM_STEPS) # Set to horizon of max episode length
	EPS = 0.05
	NUM_LABELS = 2
	WINDOW_SIZE = 14
	NUM_EPISODES = 100
	TARGET_UPDATE = 10
	RUNS = 3
	MODEL_DIR = './trained_model/'
	RESULT_DIR = './results/'

	env = img_env_orig.ImgEnv('mnist', train=True, max_steps=NUM_STEPS, channels=2, window=WINDOW_SIZE, num_labels=NUM_LABELS)
	
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
		optimizer_clf = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
		for i_episode in range(NUM_EPISODES):
			print ('run %i, episode %i'%(run, i_episode))
			total_reward_i = 0
			observation = env.reset()
			curr_label = env.curr_label.item()
			for t in range(NUM_STEPS): # allow 100 steps
				actionS, Q_values, clf_proba, action_log_probs, states = net.act(inputs=torch.from_numpy(observation).float().resize_(1, 2, 32, 32).to(device), \
					states=observation, masks=observation[1])
				actionS = actionS.cpu().numpy()[0]
				class_pred = actionS[1]
				last_observation = observation
				rand = np.random.rand()
				if rand < EPS:
					actionS = np.array(
						[np.random.choice(range(4)), np.random.choice(range(NUM_LABELS))])
				action = actionS[0]
				observation, reward, done, info = env.step(actionS)
				total_reward_i = reward + GAMMA*total_reward_i
				memory.push(torch.from_numpy(last_observation), torch.from_numpy(actionS), \
					torch.from_numpy(observation), torch.tensor([reward]).float(), torch.tensor([curr_label]))
				optimize_myNet(net, curr_label, optimizer_clf, BATCH_SIZE)

				if done:
				# print ('Done after %i steps'%(t+1))
					break

			# Update the target network, copying all weights and biases in DQN
			if i_episode % TARGET_UPDATE == 0:
				target_net.load_state_dict(net.state_dict())

			loss_classification_i = F.nll_loss(clf_proba, env.curr_label.unsqueeze_(dim=0).to(device))
			total_rewards.append(total_reward_i)
			episode_durations.append(t+1)
			loss_classification.append(loss_classification_i.detach().item())
			q_values.append(Q_values)
		run_durations.append(episode_durations)
		run_total_rewards.append(total_rewards)
		run_loss_clf.append(loss_classification)

	torch.save(net.state_dict(), os.path.join(MODEL_DIR,'model_DQN_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws.pth'.format(**locals())))
	torch.save(optimizer_clf.state_dict(), os.path.join(MODEL_DIR,'optimizer_freeze{FREEZE_CNN}_jump_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws.pth'.format(**locals())))



		
	plt.title('Class 0')
	plt.subplot(3, 1, 1)
	plt.xlabel('Episode')
	plt.ylabel('Episode_Duration')
	durations_t = torch.tensor(episode_durations[0], dtype=torch.float)
	sns.tsplot(data=run_durations, time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 1), color='red')
		
	plt.subplot(3, 1, 2)
	plt.xlabel('Episode')
	plt.ylabel('Rewards')
	total_rewards_t = torch.tensor(total_rewards[0], dtype=torch.float)
	sns.tsplot(data=run_total_rewards, time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 2), color='red')
		
	plt.subplot(3, 1, 3)
	plt.ylim(top=1)
	plt.xlabel('Episode')
	plt.ylabel('Loss Classification')
	loss_classification_t = torch.tensor(loss_classification[0], dtype=torch.float)
	sns.tsplot(data=run_loss_clf, time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 3), color='red')
	plt.savefig(os.path.join(RESULT_DIR,'DQN_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws'.format(**locals())))
	plt.show()


