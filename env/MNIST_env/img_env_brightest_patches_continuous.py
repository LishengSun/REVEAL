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
import random
import matplotlib.animation as animation





FixedCategorical = torch.distributions.Categorical

CITYSCAPE = '/datasets01/cityscapes/112817/gtFine'
IMG_ENVS = ['mnist', 'cifar10', 'cifar100', 'imagenet']


def label(axis, xy, text):
	y = xy[1] + 2  # shift y-value for label so that it's below the artist
	axis.text(xy[0]+2, y, text, ha="center", family='sans-serif', size=14)



def get_data_loader(env_id, train=True):
	kwargs = {'num_workers': 0, 'pin_memory': True}
	transform = T.Compose(
		[T.ToTensor(),
		 T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	if env_id in IMG_ENVS:
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
	metadata = {'render.modes': ['human', 'rgb_array']}
	def __init__(self, dataset, train, max_steps, channels, window=5, num_labels=10):

		self.channels = channels
		self.data_loader = get_data_loader(dataset, train=train)
		self.window = window
		self.max_steps = max_steps

		self.num_labels = num_labels
		self.num_row_choices = math.ceil(32/self.window)
		self.num_col_choices = math.ceil(32/self.window)
		self.observation_space = Box(low=0, high=1, shape=(channels, 32, 32))#shape=(channels, 32, 32))
		self.action_space = Box(np.array([0,0]), np.array([31,31]))  # x, y, both in range(0, 32)

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

		self.all_target_patches_brightness = []
		for row in range(self.num_row_choices):
			for col in range(self.num_col_choices):
				brightness = self.curr_img[0,self.window*row:self.window*(row+1),\
					self.window*col:self.window*(col+1)].numpy().mean()
				
				
				self.all_target_patches.append(row*self.num_col_choices+col)
				self.all_target_patches_brightness.append(brightness)
		self.all_target_patches, self.all_target_patches_brightness = (list(x) for x in \
			zip(*sorted(zip(self.all_target_patches, self.all_target_patches_brightness), key=lambda pair: pair[1], reverse=True)))
		
		self.targets = self.all_target_patches[:6] # top 6 brightest


		# non-initialize position at center of image, agent can decide the first position
		# self.pos = [max(0, self.curr_img.shape[1]//2-self.window), max(0, self.curr_img.shape[2]//2-self.window)]
		self.pos = [np.nan, np.nan]
		self.state = -np.ones(
			(self.channels, self.curr_img.shape[1], self.curr_img.shape[2]))

		self.num_steps = 0
		self.action_history = []
		self.pos_patch_history = []
		
		return self.state


	def step(self, action):
		done = False
		self.action_history.append(action)
		if action[0] < self.observation_space.shape[1] and action[1]< self.observation_space.shape[2]: # move
			self.pos[0] = action[0] # col
			self.pos[1] = action[1] # row
			self.curr_pos_patch = [int(self.pos[0] // self.window), \
				int(self.pos[1] // self.window)]
			if self.pos_patch_history == []:
				self.pos_patch_history = [self.curr_pos_patch]
			else:
				self.pos_patch_history.append(self.curr_pos_patch)

		else:
			print("Action out of bounds!")
			return
		self.state[0, :, :] = np.zeros(
			(1, self.curr_img.shape[1], self.curr_img.shape[2]))
		for pos_patch in self.pos_patch_history:
			self.state[0, (pos_patch[0]*self.window):((pos_patch[0]+1)*self.window), (pos_patch[1]*self.window):((pos_patch[1]+1)*self.window)] = 1
		# pdb.set_trace()
		self.state[1:,
			(self.curr_pos_patch[0]*self.window):((self.curr_pos_patch[0]+1)*self.window), (self.curr_pos_patch[1]*self.window):((self.curr_pos_patch[1]+1)*self.window)] = \
				self.curr_img[:, (self.curr_pos_patch[0]*self.window):(self.curr_pos_patch[0]+1)*self.window, (self.curr_pos_patch[1]*self.window):(self.curr_pos_patch[1]+1)*self.window]
		
		self.num_steps += 1

		action_row = int(action[0] // self.window)
		action_col = int(action[1] // self.window)

		action_brightness = self.curr_img[0,self.window*action_row:self.window*(action_row+1),\
					self.window*action_col:self.window*(action_col+1)].numpy().mean()

		max_brightness = self.all_target_patches_brightness[0]

		self.action_patch = self.curr_pos_patch[0]*self.num_col_choices + self.curr_pos_patch[1]
		done = self.action_patch in self.targets
		# print ('cost step = %f, cost brightness = %f'%(-0.5 / self.max_steps, 0.5*action_brightness/max_brightness/self.max_steps))
		reward = -1. / self.max_steps
		# reward = 0.5*(-1. / self.max_steps + (action_brightness/max_brightness) / self.max_steps)

		if done:
			reward = 1
		return self.state, reward, done, {}


	def render(self, step_i, temp_dir = './temp/', done=False, save=False, show_action_prob=False, action_prob=None, action=None):
		# inspired by https://github.com/siavashk/gym-mnist-pair/blob/master/gym_mnist_pair/envs/mnist_pair.py
		if not os.path.exists(temp_dir):
			os.makedirs(temp_dir, exist_ok=True)

		axarr1 = plt.subplot(411)
		axarr1.imshow(self.state[0, :, :], extent=[0, 32, 32, 0], vmin=0, vmax=1)
		label(axarr1, (np.clip(action[1].item(), 0, 31), np.clip(action[0].item(), 0, 31)), 'A')
		
		if done: 
			axarr1.text(16, 16, 'Done!', size=20, ha='center')
			for i, t in enumerate(self.targets):
				axarr1.add_patch(Rectangle((t%self.num_col_choices*self.window, t//self.num_col_choices*self.window),self.window,self.window, alpha=0.5, facecolor="red"))
				label(axarr1, (t%self.num_row_choices*self.window, t//self.num_row_choices*self.window), 't'+str(i))
		axarr2 = plt.subplot(412)
		axarr2.imshow(self.state[0, :, :], extent=[0, 32, 32, 0], vmin=0, vmax=1) #alpha=0.6, 
		

			
		axarr3 = plt.subplot(413)
		axarr3.imshow(self.curr_img[0,:,:], extent=[0, 32, 32, 0], vmin=0, vmax=1)
		for i, t in enumerate(self.targets):
			axarr3.add_patch(Rectangle((t%self.num_col_choices*self.window, t//self.num_col_choices*self.window),self.window,self.window, alpha=0.5, facecolor="red"))
			label(axarr3, (t%self.num_row_choices*self.window, t//self.num_row_choices*self.window), 't'+str(i))

		plt.axis('off')	
		if show_action_prob:
			axarr4 = plt.subplot(414)
			if action_prob.__class__.__name__ != 'MultivariateNormal':
				print ('Action prob of type %s not valid'%action_prob.__class__.__name__)
				return
			else:
				action_mean = action_prob.loc.detach().cpu().numpy()
				action_cov = action_prob.covariance_matrix.detach().cpu().numpy()
				x, y = np.random.multivariate_normal(action_mean, action_cov, 5000).T
				axarr4.plot(x, y, 'x')
				# label(axarr3, (15, 15), str(action_mean*31))
				# label(axarr3, (10, 10), 'A'+str(action))


		if save: plt.savefig(os.path.join(temp_dir, 'frame%i'%step_i))
		
		# print ('temp%i'%step_i)
		plt.draw()
		plt.pause(1)
		plt.close('all')






if __name__ == '__main__':
	MAX_STEPS = 49
	GAMMA = 1 - (1 / MAX_STEPS) # Set to horizon of max episode length
	temp_dir = './temp/'
	
	env = ImgEnv('mnist', train=True, max_steps=MAX_STEPS, channels=2, window=5, num_labels=10)
	env.reset()

	
	done = False
	total_reward = 0
	
	for t in range(MAX_STEPS):
		print ('t', t)
	
		action = [random.uniform(0, 32), random.uniform(0, 32)]
		
		observation, reward, done, info = env.step(action)
		# env.render(t)
		plt.imshow(observation[0,:,:])

		plt.show()

		agent_pos = env.pos
		row_move = action[0] // env.window
		col_move = action[1] % env.window
		total_reward = reward + GAMMA*total_reward

		if done: 
			print ('done after %i steps'%t)
			break


	# ani = animation.ArtistAnimation(fig_ani, ani_frames, repeat=False, interval=500)#, interval=50000, blit=True)
	# ani.save('test.mp4')

	fig_ani = plt.figure(num='test')
	ani_frames = []
	for frame_i in range(t+1):
		frame = plt.imread('temp/temp%i.png'%frame_i)
		frame = plt.imshow(frame)
		plt.axis('off')

		ani_frames.append([frame])
	ani = animation.ArtistAnimation(fig_ani, ani_frames, repeat=False, interval=500)
	ani.save('test.mp4')




