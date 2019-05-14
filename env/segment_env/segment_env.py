from gym.spaces import Discrete
import numpy as np
import matplotlib.pyplot as plt
import imageio

def generate_a_segment(length=50, noise=0.0, free_location=False):
  if free_location:
    l1 = np.random.randint(1, length-1)
    l2 = np.random.randint(1, length-1)
    left = min([l1, l2])
    right = max([l1, l2])+1
    segment = [0]*left + [1]*(right-left) + [0]*(length-right)
  else:
    right = np.random.randint(1, length)
    segment = [0]*right + [1]*(length-right)
  segment = np.array(segment) + noise * np.random.random(length)
  return segment, float(right)

def optimal_reward(segment_length, reward_pred, exploration_loss):
    state_value = np.zeros(segment_length + 1)
    state_value[1] = reward_pred
    for i in range(2, segment_length + 1):
        exploration_value = [state_value[k]*k/i + (1-k/i)*state_value[i-k] - exploration_loss for k in range(1, i)]
        state_value[i] = max(reward_pred/i, np.max(exploration_value))
    return state_value

class ImgEnv(object):
    def __init__(self, max_steps=1000, window=1, segment_length=50, noise=0.0, free_location=False,
                 expl_cost=0.05, pred_reward=1):
        self.pred_reward = pred_reward
        self.free_location = free_location
        self.noise = noise
        self.segment_length = segment_length
        self.action_space = (Discrete(self.segment_length), Discrete(self.segment_length))
        self.observation_space = Discrete(self.segment_length)
        self.window = window
        self.max_steps = max_steps
        self.to_draw = np.zeros((self.max_steps + 1, 1, self.segment_length, 3)).astype(int)
        self.expl_cost = expl_cost
        self.window = window

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, NEXT=True):
        self.pos = None
        self.num_steps = 0
        if NEXT:
            self.curr_img, self.target = generate_a_segment(length=self.segment_length, noise=self.noise,
                                                            free_location=self.free_location)

        self.state = np.zeros((2, self.segment_length)).astype(int)
        return self.state

    def get_frame(self, t, pred=None):
        true_image = np.zeros((1, self.segment_length, 3)).astype(int)
        true_image[:, self.curr_img == 1, 2] = 255
        true_image[:, self.curr_img == 0, 0] = 255

        segment_plot = np.zeros((1, self.segment_length, 3)).astype(int) + 128
        segment_plot[:, self.state[0] == 1, :] = true_image[:, self.state[0] == 1, :]
        if self.pos is not None:
            segment_plot[:, self.pos, :] = np.clip(segment_plot[:, self.pos, :] + 170, 0, 255)
        if pred is not None:
            segment_plot[:, pred, 1] = np.clip(segment_plot[:, pred, 1] + 170, 0, 255)

        self.to_draw[t, :, :, :] = segment_plot

    def render(self, e):
        true_image = np.zeros((1, self.segment_length, 3)).astype(int)
        true_image[:, self.curr_img == 1, 2] = 255
        true_image[:, self.curr_img == 0, 0] = 255

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

    def step(self, action):
        if action[0] == 0:
            self.pos = action[1]
            done = False
            reward = -self.expl_cost
            self.state[1,
            max(self.pos - self.window, 0):min(self.pos + self.window + 1, self.segment_length)] = self.curr_img[max(
                self.pos - self.window, 0):min(self.pos + self.window + 1, self.segment_length)]
            self.state[0,
            max(self.pos - self.window, 0):min(self.pos + self.window + 1, self.segment_length)] = np.ones(
                min(self.pos + self.window + 1, self.segment_length) - max(self.pos - self.window, 0)).astype(int)
            pred = None
        else:
            done = True
            reward = (self.target == action[1]).item() * self.pred_reward
            pred = action[1]

        self.get_frame(int(self.num_steps), pred=pred)
        self.num_steps += 1
        return self.state, reward, done

    def get_current_obs(self):
        return self.state

    def close(self):
        pass