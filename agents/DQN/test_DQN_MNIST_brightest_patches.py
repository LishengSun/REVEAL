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

	display = False
	NUM_STEPS = 49
	GAMMA = 1 - (1 / NUM_STEPS) # Set to horizon of max episode length
	NUM_LABELS = 2
	WINDOW_SIZE = 5
	TEST_NUM = 100
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
		done_RL = False
		done_random = False

		total_reward_t_RL = 0
		total_reward_t_random = 0
		observation_RL = env.reset()

		
		if display:
			plt.imshow(env.curr_img[0,:,:])
			currentAxis = plt.gca()
			currentAxis.add_patch(Rectangle((env.targets[0]%env.num_col_choices*env.window, env.targets[0]//env.num_col_choices*env.window),env.window,env.window, alpha=0.5, facecolor="red"))
			label((env.targets[0]%env.num_row_choices*env.window-0.5, env.targets[0]//env.num_row_choices*env.window-0.5), 't0')

			currentAxis.add_patch(Rectangle((env.targets[1]%env.num_row_choices*env.window, env.targets[1]//env.num_row_choices*env.window),env.window,env.window, alpha=0.5, facecolor="red"))
			label((env.targets[1]%env.num_row_choices*env.window-0.5, env.targets[1]//env.num_row_choices*env.window-0.5), 't1')
			
			currentAxis.add_patch(Rectangle((env.targets[2]%env.num_row_choices*env.window, env.targets[2]//env.num_row_choices*env.window),env.window,env.window, alpha=0.5, facecolor="red"))
			label((env.targets[2]%env.num_row_choices*env.window-0.5, env.targets[2]//env.num_row_choices*env.window-0.5), 't2')

			currentAxis.add_patch(Rectangle((env.targets[3]%env.num_row_choices*env.window, env.targets[3]//env.num_row_choices*env.window),env.window,env.window, alpha=0.5, facecolor="red"))
			label((env.targets[3]%env.num_row_choices*env.window-0.5, env.targets[3]//env.num_row_choices*env.window-0.5), 't3')

			currentAxis.add_patch(Rectangle((env.targets[4]%env.num_row_choices*env.window, env.targets[4]//env.num_row_choices*env.window),env.window,env.window, alpha=0.5, facecolor="red"))
			label((env.targets[4]%env.num_row_choices*env.window-0.5, env.targets[4]//env.num_row_choices*env.window-0.5), 't4')

			currentAxis.add_patch(Rectangle((env.targets[5]%env.num_row_choices*env.window, env.targets[5]//env.num_row_choices*env.window),env.window,env.window, alpha=0.5, facecolor="red"))
			label((env.targets[5]%env.num_row_choices*env.window-0.5, env.targets[5]//env.num_row_choices*env.window-0.5), 't5')

		
		ims_RL = []
		ims_random = []
		###################### test RL agent #######################
		fig_RL = plt.figure(num='RL: label = %i'%env.curr_label.item())

		observation_RL = env.reset(NEXT=False)
		for t_RL in range(NUM_STEPS):
			action, Q_values, states = RL_net.act(inputs=torch.from_numpy(observation_RL).float().resize_(1, 2, 32, 32).to(device), \
					states=observation_RL, masks=observation_RL[1])
			action = action.cpu().numpy()[0]

			observation_RL, reward_RL, done_RL, info = env.step(action[0])
			total_reward_t_RL = reward_RL + GAMMA*total_reward_t_RL
			if display:
				im_RL = plt.imshow(torch.from_numpy(observation_RL[1, :, :]), animated=True)
				ims_RL.append([im_RL])
			
			if done_RL: 
				break
		epi_len_t_RL = t_RL+1
		total_reward_RL.append(total_reward_t_RL)
		epi_len_RL.append(epi_len_t_RL)

		
		if display:
			plt.title('epi_len_RL = %i, reward_RL = %f'%(epi_len_t_RL, total_reward_t_RL))
			ani_RL = animation.ArtistAnimation(fig_RL, ims_RL, repeat=False, interval=500)#, interval=50000, blit=True)
			plt.show()
		###################### test random agent #######################
		fig_random = plt.figure(num='random: label = %i'%env.curr_label.item())
		observation_random = env.reset(NEXT=False)
		for t_rand in range(NUM_STEPS):

			action = np.array(np.random.choice(range(NUM_STEPS)))#, np.array(np.random.choice(range(16)))]
			observation_random, reward_random, done_random, info = env.step(action)
			total_reward_t_random = reward_random + GAMMA*total_reward_t_random
			if display:
				im_random = plt.imshow(torch.from_numpy(observation_random[1, :, :]), animated=True)
				ims_random.append([im_random])
			
			

			if done_random: 
				break
		epi_len_t_random = t_rand+1
		total_reward_random.append(total_reward_t_random)
		epi_len_random.append(epi_len_t_random)
		if display:
			ani_random = animation.ArtistAnimation(fig_random, ims_random, repeat=False, interval=500)#, interval=50000, blit=True)
			plt.show()

	total_reward_random_mean = np.mean(total_reward_random)
	epi_len_random_mean = np.mean(epi_len_random)
	total_reward_RL_mean = np.mean(total_reward_RL)
	epi_len_RL_mean = np.mean(epi_len_RL)


	print ('total_reward_random_mean', total_reward_random_mean)
	print ('total_reward_RL_mean', total_reward_RL_mean)
	print ('epi_len_random_mean', epi_len_random_mean)
	print ('epi_len_RL_mean', epi_len_RL_mean)



