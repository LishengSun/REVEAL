import torch
import numpy as np
import matplotlib.pyplot as plt



def value_image_from_Q_values(Q_values, window_size):
	value_image = np.zeros((32,32))-999
	i = 0
	for col in range(32 // window_size + 1):
		for row in range(32 // window_size + 1):
			value_image[window_size*col:window_size*(col+1), \
			window_size*row:window_size*(row+1)] = Q_values[0,i]
			plt.imshow(value_image)
			plt.draw()
			plt.pause(1)
			i+= 1
	return value_image

Q_values = torch.rand((1,49))

# want to have a 32 * 32 images
window_size = 5

value_image = np.zeros((32,32))+100

i = 0





plt.imshow(value_image)
plt.show()

