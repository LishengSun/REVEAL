import copy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import argparse, os
import model

from PIL import Image

from random import randint
import numpy as np
import os, time




def smoothing_average(x, factor=500):
	running_x = 0
	X = copy.deepcopy(x)
	for i in range(len(X)):
		U = 1. / min(i+1, factor)
		running_x = running_x * (1 - U) + X[i] * U
		X[i] = running_x
	return X



def generate_colors(n):
	# https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
	ret = []
	r = int(random.random() * 256)
	g = int(random.random() * 256)
	b = int(random.random() * 256)
	step = 256 / n
	for i in range(n):
		r += step
		g += step
		b += step
		r = int(r) % 256
		g = int(g) % 256
		b = int(b) % 256
		ret.append('#%02x%02x%02x' % (r, g, b))
	return ret


def value_image_from_Q_values(Q, window_size):
	value_image = np.zeros((32,32))-999
	i = 0
	for row in range(32 // window_size + 1):
		for col in range(32 // window_size + 1):
			value_image[window_size*row:window_size*(row+1), \
			window_size*col:window_size*(col+1)] = Q[0,i]
			i+= 1
	return value_image