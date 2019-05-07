import torch

from context import MNIST_env
from MNIST_env import img_env
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
from train_DQN import myNet


if __name__ == '__main__':

	
	NUM_STEPS = 10
	GAMMA = 1 - (1 / NUM_STEPS) # Set to horizon of max episode length
	NUM_LABELS = 2
	WINDOW_SIZE = 14
	TEST_NUM = 10
	MODEL_DIR = './trained_model/'
	RESULT_DIR = './results/'
	epi_len_RL = []
	total_reward_RL = []
	env = img_env.ImgEnv('mnist', train=False, max_steps=NUM_STEPS, channels=2, window=WINDOW_SIZE, num_labels=NUM_LABELS)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	

	RL_net = myNet(\
			obs_shape=env.observation_space.shape, \
			action_space=env.action_space, dataset='mnist').to(device)
	RL_state_dist = torch.load(os.path.join(MODEL_DIR, 'model_DQN_2labs_3runs_100epis_10steps_14ws.pth'), \
		map_location=lambda storage, loc: storage)
	RL_net.load_state_dict(RL_state_dist)
	# pdb.set_trace()
	for t in range(TEST_NUM):
		print ('test %i'%t)
		observation = env.reset()
		done = False
		total_reward_t_RL = 0
		epi_len_t_RL = 0
		observation_RL = env.reset()
		fig_RL = plt.figure(num='RL: label = %i'%env.curr_label.item())
		
		ims_RL = []

		while not done:
			actionS, Q_values, clf_proba, action_log_probs, states = RL_net.act(inputs=torch.from_numpy(observation).float().resize_(1, 2, 32, 32).to(device), \
					states=observation, masks=observation[1])
			actionS = actionS.cpu().numpy()[0]
			pred_label_RL = actionS[1]

			observation, reward, done, info = env.step(actionS)
			print (pred_label_RL, reward)
			im_RL = plt.imshow(torch.from_numpy(observation_RL[1, :, :]), animated=True)
			ims_RL.append([im_RL])
			total_reward_t_RL = reward + GAMMA*total_reward_t_RL
			print (total_reward_t_RL)
			print ()
			if not done:
				epi_len_t_RL += 1
		plt.title('pred_label = %i, epi_len_RL = %i, reward_RL = %f'%(pred_label_RL, epi_len_t_RL, total_reward_t_RL))
		ani_RL = animation.ArtistAnimation(fig_RL, ims_RL, repeat=False, interval=500)#, interval=50000, blit=True)
		plt.show()
