import argparse, os
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt
import random
import copy
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import seaborn as sns
import time
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from distributions import Categorical, DiagGaussian, TwoDGaussian

from context import MNIST_env
from MNIST_env import img_env_brightest_patches_continuous
from MNIST_env.img_env_brightest_patches import label


import model
import pdb
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--num_episodes', type=int, default=5000, metavar='E',
					help='number of episodes to train')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
					help='interval between training status logs (default: 100)')
parser.add_argument('--test_interval', type=int, default=1000, metavar='T',
					help='interval between testing')
args = parser.parse_args()


SavedAction = namedtuple('SavedAction', ['action', 'log_prob', 'value', 'entropy'])

def smoothing_average(x, factor=50):
	running_x = 0
	X = copy.deepcopy(x)
	for i in range(len(X)):
		U = 1. / min(i+1, factor)
		running_x = running_x * (1 - U) + X[i] * U
		X[i] = running_x
	return X


class Policy(nn.Module):
	def __init__(self, obs_shape, action_space, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
		super(Policy, self).__init__()
		self.dataset = dataset
		self.action_space = action_space
		if len(obs_shape) == 3: #our mnist case
			self.base = model.CNNBase(obs_shape[0], recurrent_policy, dataset=dataset)
		else:
			raise NotImplementedError

		if action_space.__class__.__name__ == "Box":
			num_outputs = 1 

			self.action_head = TwoDGaussian(self.base.output_size, num_outputs)
			self.value_head = nn.Linear(self.base.output_size, num_outputs)

		else:
			raise NotImplementedError

		self.saved_actions = []
		self.rewards = []



	def forward(self, x):
		raise NotImplementedError

	def act(self, inputs, states, masks, deterministic=False):
		x, _ = self.base(inputs, states, masks)
		action_prob = self.action_head(x) 
		# action_prob means in (0,1] via sigmoid, the actions will be scale to env. size later
		state_values = self.value_head(x)

		return action_prob, state_values

	def select_action(self, state):

		action_prob, state_value = self.act(inputs=torch.from_numpy(state).float().resize_(1, 2, 32, 32).to(device), \
			states=state, masks=state[1])
		action = action_prob.sample()
		action = torch.clamp(action, 1e-5, 1) # action will be in (0, 1]
		self.saved_actions.append(SavedAction(action, action_prob.log_prob(action), state_value, action_prob.entropy()))
		return action, action_prob






def finish_episode():
	R = 0
	saved_actions = policy.saved_actions
	policy_losses = []
	value_losses = []
	entropy_term = []
	returns = []
	for r in policy.rewards[::-1]: # reward from end to t0
		R = r + GAMMA * R
		returns.insert(0, R)
		
	returns = torch.tensor(returns)
	# normalize returns
	if returns.shape[0] > 1:
		eps = np.finfo(np.float32).eps.item()
		returns = (returns - returns.mean()) / (returns.std() + eps)
	else:
		returns = returns - returns

	for (action, log_prob, value, entropy), R in zip(saved_actions, returns):
		
		
		advantage = R - value.item()
		entropy_term.append(-entropy)
		policy_losses.append(-log_prob * advantage)
		value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))
	
	optimizer.zero_grad()
	p_l = torch.stack(policy_losses).sum()
	v_l = torch.stack(value_losses).sum()
	e_l = torch.stack(entropy_term).sum()
	loss = p_l + v_l + 0.1 * e_l

	loss.backward()
	optimizer.step()
	
	del policy.rewards[:]
	del policy.saved_actions[:]
	
	return p_l.item(), v_l.item(), e_l.item(), loss.item()


def test_AC_MNIST_brightest_patches_continuous(policy_net, env, num_epi, test_num):
	test_R = []
	test_len = []
	for test in range(test_num):
		state_test, ep_reward_test = env.reset(), 0
		for t_test in range(MAX_STEPS):
			
			action_test, action_prob_test = policy.select_action(state_test) # action_test in (0,1]
			action_test = action_test * env_test.action_space.high[0] # scale action
			state_test, reward_test, done_test, _ = env_test.step(action_test)
			if test < 2: # only keep 10 test videos
				env_test.render(t_test, os.path.join(RESULT_DIR_TEST, 'temp_test%i/epi%i'%(test,num_epi)), \
					done=done_test, save=True, show_action_prob=True, action_prob=action_prob_test, action=action_test)
	
			ep_reward_test = reward_test + GAMMA*ep_reward_test
			
			if done_test: 
				break
		test_R.append(ep_reward_test)
		test_len.append(t_test+1)

		if test<2:
			fig_ani = plt.figure(num='test_%i'%test)
			ani_frames = []
			for frame_i in range(1, t_test+1):
				frame = plt.imread(os.path.join(RESULT_DIR_TEST, 'temp_test%i/epi%i/frame%i.png'%(test, num_epi, frame_i)))
				plt.axis('off')
				ani_frames.append(frame)
			imageio.mimsave(os.path.join(RESULT_DIR_TEST, 'test_%i_epi%i.gif'%(test, num_epi)), ani_frames, fps=1)
		
	return test_len, test_R



if __name__ == '__main__':

	t0 = time.time()

	MAX_STEPS = 49
	WINDOW = 5
	NUM_EPISODES = args.num_episodes
	NUM_LABELS = 2
	GAMMA = 1 - (1 / MAX_STEPS) # Set to horizon of max episode length
	RESULT_DIR = './results'
	RESULT_DIR_TEST = './results/test_time'

	MODEL_DIR = './trained_model'
	RUNS = 3
	TEST_NUM = 1000


	env = img_env_brightest_patches_continuous.ImgEnv('mnist', train=True, max_steps=MAX_STEPS, channels=2, window=WINDOW, num_labels=NUM_LABELS)
	policy = Policy(env.observation_space.shape, env.action_space, dataset='mnist').to(device)
	optimizer = optim.Adam(policy.parameters(), lr=3e-3)
	
	

	run_len = []
	run_R = []
	run_1st_action_Row = [] 
	run_1st_action_Col = []
	run_policy_losses = []
	run_value_losses = []
	run_tot_loss = []

	episodes_R_test = []
	episodes_len_test = []


	for run in range(RUNS):
		episode_R = []
		episode_len = []
		episode_1st_action_Col = []
		episode_1st_action_Row = [] 
		episode_policy_losses = []
		episode_value_losses = []
		episode_tot_loss = []
		for i_episode in range(NUM_EPISODES):#count(1):
			state, ep_reward = env.reset(), 0
			for t in range(MAX_STEPS):  # No infinite loop while learning
				action, action_prob = policy.select_action(state) # action between 0 and 1, becase of action_prob
				action = action * env.action_space.high[0] # scale the action to adapt the env size
				if t == 0:
					episode_1st_action_Col.append(action[0].item())
					episode_1st_action_Row.append(action[1].item())
				state, reward, done, _ = env.step(action)
				if i_episode % args.log_interval == 0 and args.render:
					env.render(t, save = False, done=done, show_action_prob=True, action_prob=action_prob, action=action)
				
				policy.rewards.append(reward)
				ep_reward = reward + GAMMA*ep_reward
				if done:
					break

			policy_losses, value_losses, entropy_losses, tot_loss = finish_episode()
			if i_episode % args.log_interval == 0:
				print('Run {}\tEpisode {}\tLast reward: {:.2f}\tloss: {:.2f}\tv_l: {:.2f}\tp_l: {:.2f}\te_l: {:.2f}'.format(run, i_episode, ep_reward, tot_loss, value_losses, policy_losses, entropy_losses))

			episode_R.append(ep_reward)
			episode_len.append(t+1)
			episode_policy_losses.append(policy_losses)
			episode_value_losses.append(value_losses)
			episode_tot_loss.append(tot_loss)
			

			#################### save model and testing ########################
			if i_episode == 0 or (i_episode+1) % args.test_interval == 0:
				torch.save(policy.state_dict(), os.path.join(MODEL_DIR,'model_AC_MNIST_brightest_patches_continuous_%ir_%ie.pth'%(run,i_episode)))
				if run == 0: # test only 1 run
					env_test = img_env_brightest_patches_continuous.ImgEnv('mnist', train=False, max_steps=MAX_STEPS, channels=2, window=WINDOW, num_labels=NUM_LABELS)
					test_len, test_R = test_AC_MNIST_brightest_patches_continuous(policy, env_test, i_episode, TEST_NUM)
					episodes_len_test.append(test_len)
					episodes_R_test.append(test_R)

		run_len.append(episode_len)
		run_R.append(episode_R)
		run_1st_action_Col.append(episode_1st_action_Col)
		run_1st_action_Row.append(episode_1st_action_Row)
		run_policy_losses.append(episode_policy_losses)
		run_value_losses.append(episode_value_losses)
		run_tot_loss.append(episode_tot_loss)




	with open(os.path.join(RESULT_DIR,'run_len_AC_MNIST_brightest_patches_continuous_{NUM_EPISODES}e_{WINDOW}w.json'.format(**locals())), 'w') as outfile1:
		json.dump(run_len, outfile1)

	with open(os.path.join(RESULT_DIR,'run_rewards_AC_MNIST_brightest_patches_continuous_{NUM_EPISODES}e_{WINDOW}w.json'.format(**locals())), 'w') as outfile2:
		json.dump(run_R, outfile2)

	with open(os.path.join(RESULT_DIR,'run_1stCol_AC_MNIST_brightest_patches_continuous_{NUM_EPISODES}e_{WINDOW}w.json'.format(**locals())), 'w') as outfile3:
		json.dump(run_1st_action_Col, outfile3)

	with open(os.path.join(RESULT_DIR,'run_1stRow_AC_MNIST_brightest_patches_continuous_{NUM_EPISODES}e_{WINDOW}w.json'.format(**locals())), 'w') as outfile4:
		json.dump(run_1st_action_Row, outfile4)

	with open(os.path.join(RESULT_DIR,'run_polLoss_AC_MNIST_brightest_patches_continuous_{NUM_EPISODES}e_{WINDOW}w.json'.format(**locals())), 'w') as outfile5:
		json.dump(run_policy_losses, outfile5)

	with open(os.path.join(RESULT_DIR,'run_valLoss_AC_MNIST_brightest_patches_continuous_{NUM_EPISODES}e_{WINDOW}w.json'.format(**locals())), 'w') as outfile6:
		json.dump(run_value_losses, outfile6)

	with open(os.path.join(RESULT_DIR,'run_totLoss_AC_MNIST_brightest_patches_continuous_{NUM_EPISODES}e_{WINDOW}w.json'.format(**locals())), 'w') as outfile7:
		json.dump(run_tot_loss, outfile7)
	
	
	print ('total runtime = %f sec.'%(time.time()-t0))
			
	plt.subplot(211)
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	sns.tsplot(data=[smoothing_average(run_R[i]) for i in range(len(run_R))], time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(2, 1, 1), color='red', condition='train')
	sns.tsplot(data=[smoothing_average(np.array(episodes_R_test)[:, i]) for i in range(int(NUM_EPISODES/args.test_interval))], time=np.arange(0, NUM_EPISODES+args.test_interval, args.test_interval), ci=[68, 95], ax=plt.subplot(2, 1, 1), color='green', condition='test')	   


	plt.subplot(212)
	plt.xlabel('Episode')
	plt.ylabel('Episode len')
	sns.tsplot(data=[smoothing_average(run_len[i]) for i in range(len(run_len))], time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(2, 1, 2), color='red', condition='train')
	sns.tsplot(data=[smoothing_average(np.array(episodes_len_test)[:, i]) for i in range(int(NUM_EPISODES/args.test_interval))], time=np.arange(0, NUM_EPISODES+args.test_interval, args.test_interval), ci=[68, 95], ax=plt.subplot(3, 1, 2), color='green', condition='test')	   

	plt.savefig(os.path.join(RESULT_DIR, 'LearningCurve_AC_MNIST_brightest_patches_continuous_%ie'%(i_episode+1)))
	plt.show()
	plt.close()

	plt.subplot(211)
	plt.xlabel('Episode')
	plt.ylabel('1st Row')
	sns.tsplot(data=[smoothing_average(run_1st_action_Row[i]) for i in range(len(run_1st_action_Row))], time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(2, 1, 1), color='red', condition='train')


	plt.subplot(212)
	plt.xlabel('Episode')
	plt.ylabel('1st Col')
	sns.tsplot(data=[smoothing_average(run_1st_action_Col[i]) for i in range(len(run_1st_action_Col))], time=list(range(NUM_EPISODES)), ci=[68, 95], ax=plt.subplot(2, 1, 2), color='red', condition='train')
	plt.savefig(os.path.join(RESULT_DIR, 'StartAction_AC_MNIST_brightest_patches_continuous_%ie'%(i_episode+1)))
	plt.show()
	plt.close()







