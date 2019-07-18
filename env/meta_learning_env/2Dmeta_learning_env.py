import numpy as np
from meta_learning_env import metalEnv
import random

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb


class metalEnv_2D(metalEnv):
	"""docstring for metalEnv_2D"""
	def __init__(self, nb_datasets=5, *args, **kwargs):
		super(metalEnv_2D, self).__init__(*args, **kwargs)
		# super().__init__()
		self.nb_datasets = nb_datasets
		self.to_draw = np.zeros((self.max_steps + 1, self.nb_datasets, self.segment_length, 3)).astype(int)

	def generate_n_segments(self, lines=None):
		"""
		generate nb_datasets lines at a time
		"""
		if lines is None:
			if self.train:
				lines = random.sample(range(0,int(0.7*self.loss_matrix.shape[0])), self.nb_datasets)
			else:
				lines = random.sample(range(int(0.7*self.loss_matrix.shape[0])+1, self.loss_matrix.shape[0]-1), self.nb_datasets)
		noise = self.noise
		segments = self.loss_matrix.loc[lines]
		if self.use_meta_features:
			segments = np.concatenate((segments, self.metafeatures_matrix.loc[lines]), axis=1) # append metafeatures after algos
		return lines, segments + noise * np.random.random(self.segment_length)

	def reset(self, NEXT=True, lines=None):
		self.pos = None
		self.num_steps = 0
		self.total_time = 0
		self.action_history = []
		if NEXT:
			numbers, lines = self.generate_n_segments(lines)
			self.line_numbers = numbers
			self.current_lines = lines
		self.state = np.zeros((2, self.nb_datasets, self.segment_length))

		if self.use_meta_features:
			self.state[1, :, -self.nb_metafeatures:] = self.metafeatures_matrix.loc[self.line_numbers]
			self.state[0, :, -self.nb_metafeatures:] = np.ones(self.nb_metafeatures)
		return self.state


	def step(self, action):
		"""
		Compute 1 step of the game.
		Each time the agent needs to choose 1 (dataset, algo) pair
		:param action: [int in range(self.time_matrix.shape[0]), int in range(self.time_matrix.shape[1])]
		:return: array new_state, float reward, bool done
		reward = -time_cost * current_revealed_time + how much the current action helps the average cost
		"""
		print ('action = ', action)
		print ('revealed loss = %f, cost = %f'%(env.loss_matrix.iloc[env.line_numbers[action[0]], action[1]], \
					env.time_matrix.iloc[env.line_numbers[action[0]], action[1]]))
		if self.pos is None: # first step
			reward = -self.time_cost * self.time_matrix.iloc[env.line_numbers[action[0]], action[1]] \
				+ 1 - self.loss_matrix.iloc[env.line_numbers[action[0]], action[1]]
			current_average_cost = self.loss_matrix.iloc[env.line_numbers[action[0]], action[1]]
			print ('first step, reward = %f, current_average_cost = %f'%(reward, current_average_cost))
		else:
			current_best_cost = [np.min
				(self.loss_matrix.iloc[l, 
				[p[1] for p in [history for history in self.action_history] if p[0]==self.line_numbers.index(l)]]) 
				for l in self.line_numbers]
			# current_best_cost = [best_cost_dataset1, ..., best_cost_dataset5]
			current_average_cost = np.nanmean(current_best_cost)

			new_best_cost = [np.min
				(self.loss_matrix.iloc[l, 
				[p[1] for p in [history for history in self.action_history+[action]] if p[0]==self.line_numbers.index(l)]])
				for l in self.line_numbers]
			new_average_cost = np.nanmean(new_best_cost)

			reward = -self.time_cost * self.time_matrix.iloc[env.line_numbers[action[0]], action[1]] + max(0, 
				new_average_cost - current_average_cost)

			print ('current_average_cost=%f, new_average_cost=%f, reward=%f'%(current_average_cost,\
					new_average_cost, reward))

		self.pos = action

		self.state[1, self.pos[0], self.pos[1]] = self.current_lines.iloc[self.pos[0], self.pos[1]]
		self.state[0, self.pos[0], self.pos[1]] = 1.

		self.action_history.append(action)
		self.get_frame(int(self.num_steps))
		self.num_steps += 1
		done = self.num_steps == self.max_steps
		return self.state, reward, done


	# the following functions have to do with plotting the trajectory as lighting pixels in the segment
	# (as in the segment game)
	def get_frame(self, t):
		segments_plot = np.zeros((self.nb_datasets, self.segment_length, 3)).astype(int)
		for seg_i in range(self.nb_datasets):
			segments_plot[seg_i, self.state[0, seg_i, :]==0, :] += 128
			segments_plot[seg_i, self.state[0, seg_i, :]==1, 0] = \
			(self.current_lines.iloc[seg_i, self.state[0, seg_i, :]==1] * 255).astype(int)
		# segments_plot[:, self.state[0] == 0, :] = segments_plot[:, self.state[0] == 0, :] + 128
		# segments_plot[:, self.state[0] == 1, 0] = (self.current_lines[self.state[0] == 1] * 255).astype(
			# int)
		if self.pos is not None:
			segments_plot[self.pos[0], self.pos[1], :] = np.clip(segments_plot[self.pos[0], self.pos[1], :] + 170, 0, 255)
		# pdb.set_trace()
		self.to_draw[t, :, :, :] = segments_plot
		# plt.imshow(segments_plot)
		# plt.show()

	def draw(self, e):
		true_image = np.zeros((self.nb_datasets, self.segment_length, 3)).astype(int)
		true_image[:, :, 0] = (self.current_lines * 255).astype(int)

		array_list = [
			np.vstack([s_plot, true_image]) for s_plot in self.to_draw[:self.num_steps]
			]
		image_list = []
		for a in array_list:
			fig = plt.figure()
			plt.yticks([])
			plt.imshow(a)
			plt.hlines(0.5, -0.5, self.segment_length - 0.5)
			fig.canvas.draw()
			image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
			image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
			plt.close(fig)
			image_list.append(image)
		imageio.mimsave('./{}.gif'.format(e), image_list, fps=1)




if __name__ == '__main__':
	env = metalEnv_2D(use_meta_features=False)
	env.reset()
	for t in range(10):
		print ('t = ', t)
		l = random.randint(0, env.nb_datasets-1)
		c = random.randint(0, env.loss_matrix.shape[1]-1)
		action = [l, c]
		state, reward, done = env.step(action)
		print ()
		print ()
	# lines, segments = env.generate_n_segments()

	env.draw('XXX')



