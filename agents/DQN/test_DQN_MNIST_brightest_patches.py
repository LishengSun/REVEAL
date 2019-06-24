import torch

from context import MNIST_env
from MNIST_env import img_env_brightest_patches
from MNIST_env.img_env_brightest_patches import label
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import model
import torch.nn.functional as F
import torch.nn as nn
import pdb
import os
import copy
import math
from train_DQN_MNIST_brightest_patches import myNet
from matplotlib.patches import Rectangle


if __name__ == '__main__':

	render = False
	NUM_STEPS = 49
	GAMMA = 1 - (1 / NUM_STEPS) # Set to horizon of max episode length
	NUM_LABELS = 2
	WINDOW_SIZE = 5
	TEST_NUM = 20
	MODEL_DIR = './trained_model/'
	RESULT_DIR = './results/'
	epi_len_RL = []
	total_reward_RL = []

	epi_len_random = []
	total_reward_random = []
	env = img_env_brightest_patches.ImgEnv('mnist', train=False, max_steps=NUM_STEPS, channels=2, window=WINDOW_SIZE, num_labels=NUM_LABELS)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	

	RL_net = myNet(\
			obs_shape=env.observation_space.shape, \
			action_space=env.action_space, dataset='mnist').to(device)
	RL_state_dist = torch.load(os.path.join(MODEL_DIR, 'model_MNIST_brightest_patches_2labs_1runs_1000epis_49steps_5ws.pth'), \
		map_location=lambda storage, loc: storage)
	RL_net.load_state_dict(RL_state_dist)
	for t in range(TEST_NUM):
		print ('test %i'%t)

		observation = env.reset()
		
		ims_random = []

		done_RL = False
		done_random = False

		total_reward_t_RL = 0
		total_reward_t_random = 0
		


		###################### test RL agent #######################
		
		observation_RL = env.reset(NEXT=False)
		
		if render:
			env.render(0, './temp_RL/test_%i'%t, done_RL)
		for t_RL in range(NUM_STEPS):
			action, Q_values, states = RL_net.act(inputs=torch.from_numpy(observation_RL).float().resize_(1, 2, 32, 32).to(device), \
					states=observation_RL, masks=observation_RL[1])
			action = action.cpu().numpy()[0]

			observation_RL, reward_RL, done_RL, info = env.step(action[0])
			total_reward_t_RL = reward_RL + GAMMA*total_reward_t_RL
			if render:
				env.render(t_RL+1, './temp_RL/test_%i'%t, done_RL)
			
			if done_RL: 
				break
############################ just for these Chap 4 ################
		# if render:
			# fig, axarr = plt.subplots(1,1)

			# plt.imshow(env.curr_img[0,:,:], extent=[0, 32, 32, 0]) # extent = [left, right, bottom, top]

	
			# for i, ta in enumerate(env.targets):
				
			# 	axarr.add_patch(Rectangle((ta%env.num_col_choices*env.window, ta//env.num_col_choices*env.window),env.window,env.window, alpha=0.5, facecolor="red"))
			# 	label(axarr, (ta%env.num_row_choices*env.window, ta//env.num_row_choices*env.window), 't'+str(i))
			# plt.savefig(os.path.join('./temp_RL/test_%i'%t, 'target'))		
############################ just for these Chap 4 ################
		
		epi_len_t_RL = t_RL+1
		total_reward_RL.append(total_reward_t_RL)
		epi_len_RL.append(epi_len_t_RL)


		fig_ani_RL = plt.figure(num='test_%i'%t)
		ani_frames_RL = []
		for frame_RL_i in range(t_RL+2):
			frame = plt.imread('temp_RL/frame%i.png'%frame_RL_i)
			frame = plt.imshow(frame, extent=[0, 32, 32, 0])
			
			plt.axis('off')

			ani_frames_RL.append([frame])
		ani = animation.ArtistAnimation(fig_ani_RL, ani_frames_RL, repeat=False, interval=500)
		ani.save('test_RL_%i.mp4'%t)

		
		
	# 	###################### test random agent #######################
	# 	fig_random = plt.figure(num='random: label = %i'%env.curr_label.item())
	# 	observation_random = env.reset(NEXT=False)
	# 	for t_rand in range(NUM_STEPS):

	# 		action = np.array(np.random.choice(range(NUM_STEPS)))#, np.array(np.random.choice(range(16)))]
	# 		observation_random, reward_random, done_random, info = env.step(action)
	# 		total_reward_t_random = reward_random + GAMMA*total_reward_t_random
	# 		if render:
	# 			env.render(t_RL, 'temp_random')	

	# 		if done_random: 
	# 			break
	# 	epi_len_t_random = t_rand+1
	# 	total_reward_random.append(total_reward_t_random)
	# 	epi_len_random.append(epi_len_t_random)
		

	# total_reward_random_mean = np.mean(total_reward_random)
	# epi_len_random_mean = np.mean(epi_len_random)
	# total_reward_RL_mean = np.mean(total_reward_RL)
	# epi_len_RL_mean = np.mean(epi_len_RL)


	# print ('total_reward_random_mean', total_reward_random_mean)
	# print ('total_reward_RL_mean', total_reward_RL_mean)
	# print ('epi_len_random_mean', epi_len_random_mean)
	# print ('epi_len_RL_mean', epi_len_RL_mean)


	




