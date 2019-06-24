import numpy as np
import os
import torch
from gym.spaces import Discrete, Box
import torchvision.transforms as T
from torchvision import datasets
import math
import matplotlib.pyplot as plt
import copy
import pdb
from matplotlib.patches import Rectangle



FixedCategorical = torch.distributions.Categorical

CITYSCAPE = '/datasets01/cityscapes/112817/gtFine'
IMG_ENVS = ['mnist', 'cifar10', 'cifar100', 'imagenet']


def label(axis, xy, text):
    y = xy[1] + 2  # shift y-value for label so that it's below the artist
    axis.text(xy[0]+2, y, text, ha="center", family='sans-serif', size=5)



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
	def __init__(self, dataset, train, max_steps, channels, window=5, num_labels=10, num_targets=6):
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
		self.num_targets = num_targets


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

		# self.all_target_patches_percentage = []
		self.all_target_patches_brightness = []
		for row in range(self.num_row_choices):
			for col in range(self.num_col_choices):
				brightness = self.curr_img[0,self.window*row:self.window*(row+1),\
					self.window*col:self.window*(col+1)].numpy().mean()
				
				self.all_target_patches.append(row*self.num_col_choices+col)
				self.all_target_patches_brightness.append(brightness)
		self.all_target_patches, self.all_target_patches_brightness = (list(x) for x in \
			zip(*sorted(zip(self.all_target_patches, self.all_target_patches_brightness), key=lambda pair: pair[1], reverse=True)))
		
		self.targets = self.all_target_patches[:self.num_targets] # top 6 brightest
		# print ('self.target', self.target)


		# non-initialize position at center of image, agent can decide the first position
		# self.curr_pos = [max(0, self.curr_img.shape[1]//2-self.window), max(0, self.curr_img.shape[2]//2-self.window)]
		self.curr_pos = [np.nan, np.nan]
		self.pred_pos = [np.nan, np.nan]
		self.state = -np.ones(
			(self.channels, self.curr_img.shape[1], self.curr_img.shape[2]))

		self.num_steps = 0
		self.action_history = []
		self.pos_history = []
		
		return self.state

	def step(self, action):
		done = False
		self.action_history.append(action)
		if action <= self.num_row_choices * self.num_col_choices: # move

			self.curr_pos[0] = min(self.curr_img.shape[1], action // self.num_row_choices * self.window)# col move
			self.curr_pos[1] = min(self.curr_img.shape[2], action % self.num_row_choices * self.window)# row move
			if self.pos_history == []:
				self.pos_history = [copy.copy(self.curr_pos)]
			else:

				self.pos_history.append(copy.copy(self.curr_pos))
		else:
			print("Action out of bounds!")
			return
		self.state[0, :, :] = np.zeros(
			(1, self.curr_img.shape[1], self.curr_img.shape[2]))
		for pos in self.pos_history:
			self.state[0, pos[0]:pos[0]+self.window, pos[1]:pos[1]+self.window] = 1
		self.state[1:,
			self.curr_pos[0]:self.curr_pos[0]+self.window, self.curr_pos[1]:self.curr_pos[1]+self.window] = \
				self.curr_img[:, self.curr_pos[0]:self.curr_pos[0]+self.window, self.curr_pos[1]:self.curr_pos[1]+self.window]
				
		self.num_steps += 1

		# done = action[1]==self.target  or self.num_steps >= self.max_steps
		action_row = action // self.num_row_choices
		action_col = action % self.num_row_choices

		action_brightness = self.curr_img[0,self.window*action_row:self.window*(action_row+1),\
					self.window*action_col:self.window*(action_col+1)].numpy().mean()

		max_brightness = self.all_target_patches_brightness[0]

		done = action in self.targets
		# print ('cost step = %f, cost brightness = %f'%(-0.5 / self.max_steps, 0.5*action_brightness/max_brightness/self.max_steps))
		reward = -1. / self.max_steps
		# reward = 0.5*(-1. / self.max_steps + (action_brightness/max_brightness) / self.max_steps)

		if done:
			reward = 1
		return self.state, reward, done, {}

	def get_current_obs(self):
		return self.state

	def close(self):
		pass


	def render(self, step_i, temp_dir = './temp/', done=False, save=False, show_value_image=False, value_image=None):
		# inspired by https://github.com/siavashk/gym-mnist-pair/blob/master/gym_mnist_pair/envs/mnist_pair.py
		
		if not os.path.exists(temp_dir):
			os.makedirs(temp_dir, exist_ok=True)



		axarr1 = plt.subplot(411)
		axarr1.imshow(self.state[1, :, :], extent=[0, 32, 32, 0], vmin=0, vmax=1) #alpha=0.6, 
		axarr2 = plt.subplot(412)
		axarr2.imshow(self.state[0, :, :], extent=[0, 32, 32, 0], vmin=0, vmax=1) #alpha=0.6, 
		# label(axarr2, (self.curr_pos[1], self.curr_pos[0]), step_i)

		plt.axis('off')	
		
		if done: 
			axarr1.text(16, 16, 'Done!', size=20, ha='center')
			for i, t in enumerate(self.targets):
			
				axarr1.add_patch(Rectangle((t%self.num_col_choices*self.window, t//self.num_col_choices*self.window),self.window,self.window, alpha=0.5, facecolor="red"))
				label(axarr1, (t%self.num_row_choices*self.window, t//self.num_row_choices*self.window), 't'+str(i))

		axarr3 = plt.subplot(413)
		axarr3.imshow(self.curr_img[0,:,:], extent=[0, 32, 32, 0], vmin=0, vmax=1)
		for i, t in enumerate(self.targets):
			
			axarr3.add_patch(Rectangle((t%self.num_col_choices*self.window, t//self.num_col_choices*self.window),self.window,self.window, alpha=0.5, facecolor="red"))
			label(axarr3, (t%self.num_row_choices*self.window, t//self.num_row_choices*self.window), 't'+str(i))

		plt.axis('off')	
		if show_value_image:
			axarr4 = plt.subplot(414)
			mappable = axarr4.imshow(value_image)#, vmin=0, vmax=1)
			plt.colorbar(mappable)#im, cax=cax, orientation='horizontal')
			label(axarr4, (self.curr_pos[1], self.curr_pos[0]), 'A')
			# axarr3.imshow(self.state[0, :, :], extent=[0, 32, 32, 0], vmin=0, vmax=1, alpha=0.4)
			plt.axis('off')	
		if save: plt.savefig(os.path.join(temp_dir, 'frame%i'%step_i), bbox_inches='tight')	
		
		plt.draw()
		plt.pause(0.5)
		plt.close('all')


if __name__ == '__main__':
	MAX_STEPS = 49
	GAMMA = 1 - (1 / MAX_STEPS) # Set to horizon of max episode length

	env = ImgEnv('mnist', train=True, max_steps=MAX_STEPS, channels=2, window=5, num_labels=10)
	env.reset()


	done = False
	total_reward = 0
	for t in range(MAX_STEPS):
		action = np.array(np.random.choice(range(49)))#, np.array(np.random.choice(range(16)))]
		observation, reward, done, info = env.step(action)
		env.render(t, temp_dir = './temp/', done=done, save=False, show_value_image=True, value_image=observation[0,:,:])
		

		if done: 
			break

