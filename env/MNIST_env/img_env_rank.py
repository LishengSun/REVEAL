import numpy as np

import torch
from gym.spaces import Discrete, Box
import torchvision.transforms as T
from torchvision import datasets
import math
import matplotlib.pyplot as plt
import copy
import pdb


FixedCategorical = torch.distributions.Categorical

CITYSCAPE = '/datasets01/cityscapes/112817/gtFine'
IMG_ENVS = ['mnist', 'cifar10', 'cifar100', 'imagenet']


def get_data_loader(env_id, train=True):
	kwargs = {'num_workers': 0, 'pin_memory': True}
	transform = T.Compose(
		[T.ToTensor(),
		 T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	if env_id in IMG_ENVS:
		# print (env_id)
		if env_id == 'mnist':
			transform = T.Compose([
						   T.Resize(size=(32, 32)),
						   T.ToTensor(),
						   T.Normalize((0.1307,), (0.3081,))
					   ])
			dataset = datasets.MNIST
		elif env_id == 'cifar10':
			dataset = datasets.CIFAR10
		elif env_id == 'cifar100':
			dataset = datasets.CIFAR100
		elif env_id == 'imagenet':
			normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

			if train:
				data_dir = ''
			else:
				data_dir = ''
			dataset = datasets.ImageFolder(
				data_dir,
				transforms.Compose([
					transforms.RandomResizedCrop(224),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					normalize,
				]))
		loader = torch.utils.data.DataLoader(
			dataset('data', train=train, download=True,
					transform=transform),
			batch_size=1, shuffle=True, **kwargs)
	return loader


class ImgEnv(object):
	def __init__(self, dataset, train, max_steps, channels, window=5, num_labels=10):
		# Jump action space has 28/window*28/window+1 actions: 

		self.channels = channels
		self.data_loader = get_data_loader(dataset, train=train)
		self.window = window
		self.max_steps = max_steps

		self.num_labels = num_labels
		self.num_row_choices = math.ceil(32/self.window)
		self.num_col_choices = math.ceil(32/self.window)

		# self.action_space = [Discrete(self.num_row_choices), Discrete(self.num_col_choices), Discrete(2)]
		self.action_space = Discrete(self.num_row_choices*self.num_col_choices) # done detected automatically
		self.observation_space = Box(low=0, high=1, shape=(channels, 32, 32))#shape=(channels, 32, 32))
		


	def seed(self, seed):
		np.random.seed(seed)

	def reset(self, NEXT=True):
		if NEXT: # whether switch to next image
			self.curr_img, self.curr_label = next(iter(self.data_loader))
			while self.curr_label >= self.num_labels:
				self.curr_img, self.curr_label = next(iter(self.data_loader))
			self.curr_img = self.curr_img.squeeze(0)
			self.curr_label = self.curr_label.squeeze(0)

		# identify patches to be revealed
		self.all_target_patches = []
		self.all_target_patches_percentage = []
		for row in range(self.num_row_choices):
			for col in range(self.num_col_choices):
				# pdb.set_trace()
				# percent = (self.curr_img[0,self.window*row:self.window*(row+1),\
					# self.window*col:self.window*(col+1)].numpy()>0).sum() / (self.window**2)
				percent = self.curr_img[0,self.window*row:self.window*(row+1),\
					self.window*col:self.window*(col+1)].numpy().mean()
				self.all_target_patches.append(row*self.num_col_choices+col)
				self.all_target_patches_percentage.append(percent)
		self.all_target_patches, self.all_target_patches_percentage = (list(x) for x in \
			zip(*sorted(zip(self.all_target_patches, self.all_target_patches_percentage), key=lambda pair: pair[1], reverse=True)))
		# print (self.all_target_patches, self.all_target_patches_percentage)
		self.target = self.all_target_patches[0]
		# print ('self.target', self.target)


		# non-initialize position at center of image, agent can decide the first position
		# self.pos = [max(0, self.curr_img.shape[1]//2-self.window), max(0, self.curr_img.shape[2]//2-self.window)]
		self.pos = [np.nan, np.nan]
		self.pred_pos = [np.nan, np.nan]
		self.state = -np.ones(
			(self.channels, self.curr_img.shape[1], self.curr_img.shape[2]))

		self.num_steps = 0
		
		return self.state

	def step(self, action):
		done = False
		if action[0] <= self.num_row_choices * self.num_col_choices: # move
				self.pos[0] = min(self.curr_img.shape[1], action[0] // self.num_col_choices * self.window)# row move
				self.pos[1] = min(self.curr_img.shape[2], action[0] % self.num_col_choices * self.window)# col move

				# self.pred_pos[0] = min(self.curr_img.shape[1], self.window*(action[1] // self.num_col_choices))
				# self.pred_pos[1] = min(self.curr_img.shape[2], self.window*(action[1] % self.num_col_choices))
		else:
			print("Action out of bounds!")
			return
		self.state[0, :, :] = np.zeros(
			(1, self.curr_img.shape[1], self.curr_img.shape[2]))
		self.state[0, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = 1
		self.state[1:,
			self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window] = \
				self.curr_img[:, self.pos[0]:self.pos[0]+self.window, self.pos[1]:self.pos[1]+self.window]
		# self.state[1:,
			# self.pred_pos[0]:self.pred_pos[0]+self.window, self.pred_pos[1]:self.pred_pos[1]+self.window] = 20
				
		self.num_steps += 1

		done = action[1]==self.target  or self.num_steps >= self.max_steps
		reward = - 1. / self.max_steps
		if done and action[1]==self.target:
			reward = 1
		return self.state, reward, done, {}

	def get_current_obs(self):
		return self.state

	def close(self):
		pass

if __name__ == '__main__':
	MAX_STEPS = 16
	GAMMA = 1 - (1 / MAX_STEPS) # Set to horizon of max episode length

	env = ImgEnv('mnist', train=True, max_steps=MAX_STEPS, channels=2, window=8, num_labels=10)
	env.reset()
	plt.imshow(env.curr_img[0,:,:])
	plt.show()
	# fig = plt.figure()
	
	done = False
	total_reward = 0
	for t in range(MAX_STEPS):
	# while not done:
		# action = [0, env.target]
		action = [np.array(np.random.choice(range(16))), np.array(np.random.choice(range(16)))]
		observation, reward, done, info = env.step(action)
		agent_pos = env.pos
		row_move = action[0] // env.num_col_choices
		col_move = action[0] % env.num_col_choices
		total_reward = reward + GAMMA*total_reward
		plt.imshow(observation[1,:,:])
		plt.title('a=%i,%i, target=%i, r=%f, total_r = %f'%(action[0], action[1],env.target, reward, total_reward))
		plt.show()

		if done: 
			break

