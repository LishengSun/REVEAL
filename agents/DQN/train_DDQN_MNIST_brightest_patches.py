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
from MNIST_env import img_env_brightest_patches
import utils
from visualization_utils import *
import seaborn as sns
import imageio
import argparse, os



import model

from PIL import Image

from random import randint
import numpy as np
import os, time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")








def soft_update(target_net, source_net, tau=0.1):
	# Adapt from https://github.com/LishengSun/REVEAL/blob/master/agents/Double_DQN/Double_DQN.py
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    return target_net



class myNet(nn.Module):

	"""
	Q value nets, this is used both for Q network and target network
	"""
	def __init__(self, obs_shape, action_space, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
		super(myNet, self).__init__()
		self.dataset = dataset
		if len(obs_shape) == 3: #our mnist case
			self.base = model.CNNBase(obs_shape[0], recurrent_policy, dataset=dataset)
		else:
			raise NotImplementedError

		if action_space.__class__.__name__ == "Discrete": # our case
			num_outputs = action_space.n
			self.dist = Categorical(self.base.output_size, num_outputs)
		else:
			raise NotImplementedError


		self.state_size = self.base.state_size

	def forward(self, inputs, states, masks):
		raise NotImplementedError

	def act(self, inputs, states, masks, deterministic=False):
		actor_features, states = self.base(inputs, states, masks)
		self.actor_features = actor_features

		Q_values, dist = self.dist(actor_features)
		
		if deterministic:
			action = dist.mode()
		else:
			try:
				action = dist.sample()
			except Exception:
				pdb.set_trace()


		return action, Q_values, dist, states #dist.logits = Q values



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



def optimize_myNet(net, optimizer, BATCH_SIZE=128):
	if len(memory) < BATCH_SIZE:
		return
	# sample state_batch from memory
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
	# print ('reward_batch', reward_batch)
	
	# use Q_net to predict action given sampled state_batch
	net_action, Q_values_batch, _, _ = net.act(
		inputs=state_batch.float(),
		states=state_batch, masks=state_batch[1])
	
	state_action_values = Q_values_batch.gather(1, action_batch[:,0].view(BATCH_SIZE,1))
	

	# use target net to evaluate action
	_, next_Q_values_batch, _, _= target_net.act(\
		inputs=non_final_next_states.float(),\
		states=non_final_next_states, masks=non_final_next_states[1])
	# print ('from target net: next_Q_values_batch', next_Q_values_batch)

	net_next_action, _, _, _= net.act(\
		inputs=non_final_next_states.float(),\
		states=non_final_next_states, masks=non_final_next_states[1])


	# Double DQN, instead of using target_net_Q(target_net_next_action) to compute expected values,
	# use target_net_Q(net_next_action)
	ss = 0
	next_state_values = torch.zeros(BATCH_SIZE).to(device)
	for s in range(len(non_final_mask)):
		if non_final_mask[s] == 1: # non final state
			next_state_values[s] = next_Q_values_batch[ss, net_next_action[ss]]
			ss+=1

	# next_state_values[non_final_mask] = next_Q_values_batch.max(1)[0].detach()
	# pdb.set_trace()
	
	expected_state_action_values = next_state_values * GAMMA + reward_batch # Compute the expected Q values

	loss_navig = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
	
	total_loss = loss_navig  #only train on navigation loss

	optimizer.zero_grad()
	total_loss.backward()
	# pdb.set_trace()
	# for name, param in net.named_parameters():
		# print(name, torch.max(torch.abs(param.grad)).data if param.grad is not None else None)
	for param in filter(lambda p: p.requires_grad, net.parameters()):
		param.grad.data.clamp_(-1, 1) #gradient clipping prevent only the exploding gradient problem

	
	optimizer.step()

	return total_loss



def test_DDQN_MNIST_brightest_patches(Qnet, env, num_epi, TEST_NUM):
	test_R = []
	test_len = []
	for test in range(TEST_NUM):
		ep_reward_test = 0
		observation_test = env.reset()
		
		for t_test in range(env.max_steps):
			action_test, Q_values_test, dist_test, states_test = Qnet.act(inputs=torch.from_numpy(observation_test).float().resize_(1, 2, 32, 32).to(device), \
					states=observation_test, masks=observation_test[1])
			# print ('action_test', action_test)
			if device.type != 'cpu':
				action_test = action_test.cpu()
			action_test = action_test.numpy()[0]
			
			# pdb.set_trace()
			observation_test, reward_test, done_test, _ = env.step(action_test[0])
			ep_reward_test = reward_test + ep_reward_test*GAMMA

			if test<10: # save 10 test video frames
				Qvalue_image_test = value_image_from_Q_values(dist_test.probs.detach().cpu(), env.window)
				env.render(step_i=t_test+1, temp_dir=os.path.join(RESULT_DIR_TEST, 'temp_test%i/epi%i'%(test,num_epi)), \
				done=done_test, save=True, show_value_image=True, value_image=Qvalue_image_test)
			if done_test:
				break


		test_len.append(t_test+1)
		test_R.append(ep_reward_test)

		if test<10: # create 10 videos
			fig_ani = plt.figure(num='test_%i'%test)
			ani_frames = []
			for frame_i in range(1, t_test+2):
				frame = plt.imread(os.path.join(RESULT_DIR_TEST, 'temp_test%i/epi%i/frame%i.png'%(test, num_epi, frame_i)))
		
				plt.axis('off')

				ani_frames.append(frame)
			imageio.mimsave(os.path.join(RESULT_DIR_TEST, 'test_%i_epi%i.gif'%(test, num_epi)), ani_frames, fps=1)
		
	return test_len, test_R



####################### TRAINING and TESTING ############################
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
	parser.add_argument('--num_episodes', type=int, default=2000, metavar='E',
						help='number of episodes to train')
	parser.add_argument('--render', action='store_true',
						help='render the environment')
	parser.add_argument('--log_interval', type=int, default=100, metavar='N',
						help='interval between training status logs (default: 100)')
	parser.add_argument('--test', action='store_true',
						help='Whether to run test')
	parser.add_argument('--test_num', type=int, default=1000, 
						help='Number of test instances to run')
	parser.add_argument('--test_interval', type=int, default=500, metavar='T',
						help='interval between testing')
	parser.add_argument('--num_runs', type=int, default=3, 
						help='number of runs')
	parser.add_argument('--learning_rate_decay', type=float, default=0, 
						help='used in optim scheduler')
	parser.add_argument('--epsilon_start', type=int, default=0.2, 
						help='Chance to sample a random action when taking an action. \
						Epsilon is decayed over time to epsilon_end, and this is the start value')
	parser.add_argument('--epsilon_end', type=int, default=0.05, 
						help='Chance to sample a random action when taking an action. \
						Epsilon is decayed over time to epsilon_end, and this is the end value')

	args = parser.parse_args()

	t0 = time.time()
	BATCH_SIZE = 128
	NUM_STEPS = 49
	GAMMA = 1 - (1 / NUM_STEPS) # Set to horizon of max episode length
	
	NUM_LABELS = 2
	WINDOW_SIZE = 5
	NUM_EPISODES = args.num_episodes
	EPS_START = args.epsilon_start
	EPS_END = args.epsilon_end
	EPS = np.linspace(EPS_START, EPS_END, NUM_EPISODES)
	
	TARGET_UPDATE = 10
	RUNS = args.num_runs
	MODEL_DIR = './trained_model/'
	RESULT_DIR = './results/'
	RESULT_DIR_TEST = './results/test_time'
	RENDER = args.render

	TEST = args.test
	TEST_NUM = args.test_num
	TEST_INTERVAL = args.test_interval

	env = img_env_brightest_patches.ImgEnv('mnist', train=True, max_steps=NUM_STEPS, channels=2, window=WINDOW_SIZE, num_labels=NUM_LABELS)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	run_durations = []
	run_total_rewards = []
	run_loss = []

	episode_R_test = []
	episode_len_test = []
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
		loss = []
		q_values = []
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005, amsgrad=True)
		q_distribution_checkpoints = int(NUM_EPISODES / 10)
		q_colors = generate_colors(NUM_EPISODES//q_distribution_checkpoints)

		for i_episode in range(NUM_EPISODES):
			total_reward_i = 0
			observation = env.reset()
			epsilon = EPS[i_episode]
			# EPS -= EPS_annealing_rate
			# print ('run %i, episode %i, exploration eps=%f'%(run, i_episode, EPS))

			for t in range(NUM_STEPS): # allow 100 steps
				action, Q_values, dist, states = net.act(inputs=torch.from_numpy(observation).float().resize_(1, 2, 32, 32).to(device), \
					states=observation, masks=observation[1])
				# pdb.set_trace()
				if device.type != 'cpu':
					action = action.cpu()
				action = action.numpy()[0]
				current_state = observation
				rand = np.random.rand()
				if rand < epsilon:
					action = np.array([np.random.choice(range(env.action_space.n))])
				observation, reward, done, info = env.step(action[0])
				if RENDER and i_episode % 50 == 0: # only render every 50 episodes during training
					# Qvalue_image = value_image_from_Q_values(dist.probs.detach().cpu(), env.window)
					Qvalue_image = value_image_from_Q_values(Q_values.detach().cpu(), env.window)
					action_target, Q_values_target, dist_target, states_target = target_net.act(inputs=torch.from_numpy(observation).float().resize_(1, 2, 32, 32).to(device), \
						states=observation, masks=observation[1])
					target_Qvalue_image = value_image_from_Q_values(Q_values_target.detach().cpu(), env.window)
					env.render(t+1, os.path.join(RESULT_DIR, 'temp_train/epi%i'%(i_episode)), \
						done=done, save=True, show_value_image=True, value_image=Qvalue_image, value_image_target=target_Qvalue_image)

				total_reward_i = reward + GAMMA*total_reward_i
				if not done:
					next_state = observation
					# print ('next_state = 1')
					memory.push(torch.from_numpy(current_state), torch.from_numpy(action), \
					torch.from_numpy(next_state), torch.tensor([reward]).float())
				else:
					next_state = None
					# print ('next_state', next_state)
					memory.push(torch.from_numpy(current_state), torch.from_numpy(action), \
					next_state, torch.tensor([reward]).float())
				loss_i = optimize_myNet(net, optimizer, BATCH_SIZE)
				
				if done:
					# print ('Done after %i steps, reward = %f'%(t+1, total_reward_i))
					break

			print ('run %i, e %i, After %i steps, reward = %f'%(run, i_episode, t+1, total_reward_i))
			
			# Update the target network, copying all weights and biases in DQN
			if i_episode % TARGET_UPDATE == 0:
				target_net.load_state_dict(net.state_dict())
				# target_net = soft_update(target_net, net)

			total_rewards.append(total_reward_i)
			try:
				loss.append(loss_i.item())
			except Exception: # at the beginning of training, memory len still < batch_size
				pass
			episode_durations.append(t+1)
			q_values.append(Q_values)

			if (i_episode+1) % q_distribution_checkpoints == 0:
				plt.plot(Q_values.cpu().detach().numpy()[0], c=q_colors[(i_episode+1)//q_distribution_checkpoints-1], label='episode %i'%i_episode)
				plt.title('Q value distributions')
				plt.legend()
				plt.savefig(os.path.join(RESULT_DIR,'Q_MNIST_brightest_patches_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws'.format(**locals())))



			########################### test ###############################

			if TEST and (i_episode == 0 or (i_episode+1) % TEST_INTERVAL == 0):

				print ('run %i, episode %i: done after %i steps, reward =%f'%(run, i_episode, t+1, total_reward_i))
				torch.save(net.state_dict(), os.path.join(MODEL_DIR,'DDQN_MNIST_brightest_patches_{NUM_LABELS}labs_{run}runs_{i_episode}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws.pth'.format(**locals())))
				torch.save(optimizer.state_dict(), os.path.join(MODEL_DIR,'optimDDQN_MNIST_brightest_patches_{NUM_LABELS}labs_{run}runs_{i_episode}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws.pth'.format(**locals())))


				if run == 0: # test one run
					env_test = img_env_brightest_patches.ImgEnv('mnist', train=False, max_steps=NUM_STEPS, channels=2, window=WINDOW_SIZE, num_labels=NUM_LABELS)

					test_len, test_R = test_DDQN_MNIST_brightest_patches(net, env_test, i_episode, TEST_NUM)

					episode_len_test.append(test_len)
					episode_R_test.append(test_R)


		run_durations.append(episode_durations)
		run_total_rewards.append(total_rewards)
		run_loss.append(loss)


	with open(os.path.join(RESULT_DIR,'run_len_DDQN_MNIST_brightest_patches_{NUM_EPISODES}e_{WINDOW_SIZE}w.json'.format(**locals())), 'w') as outfile1:
		json.dump(run_durations, outfile1)

	with open(os.path.join(RESULT_DIR,'run_rewards_DDQN_MNIST_brightest_patches_{NUM_EPISODES}e_{WINDOW_SIZE}w.json'.format(**locals())), 'w') as outfile2:
		json.dump(run_total_rewards, outfile2)


	with open(os.path.join(RESULT_DIR,'run_loss_DDQN_MNIST_brightest_patches_{NUM_EPISODES}e_{WINDOW_SIZE}w.json'.format(**locals())), 'w') as outfile7:
		json.dump(run_loss, outfile7)

	
	print ('total runtime = %f sec.'%(time.time()-t0))
			

	
	
	plt.subplot(211)
	plt.xlabel('Episode')
	plt.ylabel('Episode_Duration')
	sns.tsplot(data=[smoothing_average(run_durations[i]) for i in range(len(run_durations))], time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(2, 1, 1), color='red', condition='train')
	if TEST:
		sns.tsplot(data=[smoothing_average(np.array(episode_len_test)[:, i]) for i in range(TEST_NUM)], time=np.arange(0, NUM_EPISODES+TEST_INTERVAL, TEST_INTERVAL), ci=[68, 95], ax=plt.subplot(2, 1, 1), color='green', condition='test')	   
	plt.plot(np.linspace(0, NUM_EPISODES, 100), [1]*100, label='optimal')

	plt.subplot(212)
	plt.xlabel('Episode')
	plt.ylabel('Rewards')
	sns.tsplot(data=[smoothing_average(run_total_rewards[i]) for i in range(len(run_total_rewards))], time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(2, 1, 2), color='red', condition='train')
	if TEST:
		sns.tsplot(data=[smoothing_average(np.array(episode_R_test)[:, i]) for i in range(TEST_NUM)], time=np.arange(0, NUM_EPISODES+TEST_INTERVAL, TEST_INTERVAL), ci=[68, 95], ax=plt.subplot(2, 1, 2), color='green', condition='test')
	plt.plot(np.linspace(0, NUM_EPISODES, 100), [1]*100, label='optimal')
	plt.legend()
	plt.savefig(os.path.join(RESULT_DIR,'MNIST_DDQN_brightest_patches_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws'.format(**locals())))
	plt.show()

	

	# plt.subplot(413)
	# plt.xlabel('Episode')
	# plt.ylabel('max Q')
	# q_value_max = [np.max(q_values[i].cpu().detach().numpy(), axis=1) for i in range(NUM_EPISODES)]
	# plt.plot(smoothing_average(q_value_max))

	# plt.subplot(414)
	# plt.xlabel('Episode')
	# plt.ylabel('Navig Loss from optimizer')
	# plt.plot(range(NUM_EPISODES-len(run_loss[0]), NUM_EPISODES), smoothing_average(run_loss[0]))
	# plt.savefig(os.path.join(RESULT_DIR,'MNIST_brightest_patches_{NUM_LABELS}labs_{RUNS}runs_{NUM_EPISODES}epis_{NUM_STEPS}steps_{WINDOW_SIZE}ws'.format(**locals())))
	# plt.show()
		# sns.tsplot(data=smoothing_average(q_value_max), time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(3, 1, 2), color='red')

	# plt.subplot(514)
	# plt.xlabel('Episode')
	# plt.ylabel('Nvg loss')
	# plt.plot(smoothing_average(loss_navig[50:], factor=100))


	# plt.subplot(515)
	# plt.xlabel('Episode')
	# plt.ylabel('Clf loss')
	# plt.plot(smoothing_average(loss_clf[50:], factor=100))


