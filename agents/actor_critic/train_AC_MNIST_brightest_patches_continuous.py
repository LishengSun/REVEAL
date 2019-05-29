import argparse, os
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt
import random
import copy
from matplotlib.patches import Rectangle
import matplotlib.animation as animation



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from distributions import Categorical, DiagGaussian, TwoDGaussian

from context import MNIST_env
from MNIST_env import img_env_brightest_patches_continuous
from MNIST_env.img_env_brightest_patches import label


import model
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 100)')
parser.add_argument('--test_interval', type=int, default=1000, metavar='N',
                    help='interval between testing video (default: 1000)')
args = parser.parse_args()

MAX_STEPS = 49
WINDOW = 5
EPS = 0.05
NUM_EPISODES = 5000
EPS_annealing_rate = (EPS-0.05) / NUM_EPISODES # annealed to 0.05 at the end of episodes
NUM_LABELS = 2
GAMMA = 1 - (1 / MAX_STEPS) # Set to horizon of max episode length
RESULT_DIR = '/Users/lishengsun/Dropbox/OnGoingWork/REVEAL/agents/actor_critic/results'
MODEL_DIR = '/Users/lishengsun/Dropbox/OnGoingWork/REVEAL/agents/actor_critic/trained_model'
TEST_NUM = 100


env = img_env_brightest_patches_continuous.ImgEnv('mnist', train=True, max_steps=MAX_STEPS, channels=2, window=WINDOW, num_labels=NUM_LABELS)



SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def smoothing_average(x, factor=50):
    running_x = 0
    X = copy.deepcopy(x)
    for i in range(len(X)):
        U = 1. / min(i+1, factor)
        running_x = running_x * (1 - U) + X[i] * U
        X[i] = running_x
    return X


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, recurrent_policy=False, dataset=None, resnet=False, pretrained=False):
        super(Policy, self).__init__()
        self.dataset = dataset
        if len(obs_shape) == 3: #our mnist case
            self.base = model.CNNBase(obs_shape[0], recurrent_policy, dataset=dataset)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Box":
            num_outputs = 1 

            self.action_head = TwoDGaussian(self.base.output_size, num_outputs)
            self.value_head = nn.Linear(self.base.output_size, num_outputs)

        else:
            raise NotImplementedError

        self.saved_actions = []
        self.rewards = []



    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        x, _ = self.base(inputs, states, masks)
        action_prob = self.action_head(x)
        state_values = self.value_head(x)
        # print ('state_value: ', state_values)

        return action_prob, state_values


policy = Policy(env.observation_space.shape, env.action_space, dataset='mnist')
optimizer = optim.Adam(policy.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()



      

def select_action(state, env):
    # state = torch.from_numpy(state).float()
    action_prob, state_value = policy.act(inputs=torch.from_numpy(state).float().resize_(1, 2, 32, 32).to(device), \
                    states=state, masks=state[1])
    action = action_prob.sample()
    action = torch.clamp(action, env.action_space.low[0], env.action_space.high[0])

    # print ('action before', action)
    
    rand = np.random.rand()
    if rand < EPS:
        action = torch.FloatTensor(2).uniform_(0,31)
    # pdb.set_trace()
    policy.saved_actions.append(SavedAction(action_prob.log_prob(action), state_value))
    return action



def finish_episode():
    R = 0
    saved_actions = policy.saved_actions
    policy_losses = []
    value_losses = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        returns.insert(0, R)
        # if R == 1:
        # print ('R', R)
    returns = torch.tensor(returns)
    if returns.shape[0] > 1:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    for (log_prob, value), R in zip(saved_actions, returns):
        # print ('(log_prob, value), R', (log_prob, value), R)
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    
    loss.backward()
    optimizer.step()
    # print ('========optimized===========')
    # print ('policy.rewards', np.min(policy.rewards), np.max(policy.rewards))
    del policy.rewards[:]
    del policy.saved_actions[:]
    p_l = [p.detach().item() for p in policy_losses]
    v_l = [v.detach().item() for v in value_losses]
    return p_l, v_l, loss.detach().item()





if __name__ == '__main__':
    episode_R = []
    running_reward = 0
    episode_len = []
    episode_1st_action_Col = []
    episode_1st_action_Row = [] 
    episode_policy_losses = []
    episode_value_losses = []
    episode_tot_loss = []
    for i_episode in range(NUM_EPISODES):#count(1):
        # print ('episode %i'%i_episode)
        state, ep_reward = env.reset(), 0
        for t in range(MAX_STEPS):  # Don't infinite loop while learning
            action = select_action(state, env)
            if t == 0:
                episode_1st_action_Col.append(action[0])
                episode_1st_action_Row.append(action[1])
            state, reward, done, _ = env.step(action)
            
            policy.rewards.append(reward)
            ep_reward = reward + GAMMA*ep_reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        policy_losses, value_losses, tot_loss = finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        episode_R.append(ep_reward)
        episode_len.append(t+1)
        episode_policy_losses.append(np.mean(policy_losses))
        episode_value_losses.append(np.mean(value_losses))
        episode_tot_loss.append(tot_loss)
        

        if i_episode % args.test_interval == 0:
            torch.save(policy.state_dict(), os.path.join(MODEL_DIR,'model_AC_MNIST_brightest_patches_continuous_%ie.pth'%i_episode))
            ######### test video #######
            ims_test = []
            fig_test = plt.figure(num='Testing after %i epi of training'%i_episode)

            env_test = img_env_brightest_patches_continuous.ImgEnv('mnist', train=False, max_steps=MAX_STEPS, channels=2, window=WINDOW, num_labels=NUM_LABELS)
            state_test, ep_reward_test = env_test.reset(), 0
            im_target = plt.imshow(env_test.curr_img[0,:,:])
            currentAxis = plt.gca()
            currentAxis.add_patch(Rectangle((env_test.targets[0]%env_test.num_col_choices*env_test.window, env_test.targets[0]//env_test.num_col_choices*env_test.window),env_test.window,env_test.window, alpha=0.5, facecolor="red"))
            label((env_test.targets[0]%env_test.num_row_choices*env_test.window, env_test.targets[0]//env_test.num_row_choices*env_test.window), 't0')

            currentAxis.add_patch(Rectangle((env_test.targets[1]%env_test.num_row_choices*env_test.window, env_test.targets[1]//env_test.num_row_choices*env_test.window),env_test.window,env_test.window, alpha=0.5, facecolor="red"))
            label((env_test.targets[1]%env_test.num_row_choices*env_test.window, env_test.targets[1]//env_test.num_row_choices*env_test.window), 't1')

            currentAxis.add_patch(Rectangle((env_test.targets[2]%env_test.num_row_choices*env_test.window, env_test.targets[2]//env_test.num_row_choices*env_test.window),env_test.window,env_test.window, alpha=0.5, facecolor="red"))
            label((env_test.targets[2]%env_test.num_row_choices*env_test.window, env_test.targets[2]//env_test.num_row_choices*env_test.window), 't2')

            currentAxis.add_patch(Rectangle((env_test.targets[3]%env_test.num_row_choices*env_test.window, env_test.targets[3]//env_test.num_row_choices*env_test.window),env_test.window,env_test.window, alpha=0.5, facecolor="red"))
            label((env_test.targets[3]%env_test.num_row_choices*env_test.window, env_test.targets[3]//env_test.num_row_choices*env_test.window), 't3')

            currentAxis.add_patch(Rectangle((env_test.targets[4]%env_test.num_row_choices*env_test.window, env_test.targets[4]//env_test.num_row_choices*env_test.window),env_test.window,env_test.window, alpha=0.5, facecolor="red"))
            label((env_test.targets[4]%env_test.num_row_choices*env_test.window, env_test.targets[4]//env_test.num_row_choices*env_test.window), 't4')

            currentAxis.add_patch(Rectangle((env_test.targets[5]%env_test.num_row_choices*env_test.window, env_test.targets[5]//env_test.num_row_choices*env_test.window),env_test.window,env_test.window, alpha=0.5, facecolor="red"))
            label((env_test.targets[5]%env_test.num_row_choices*env_test.window, env_test.targets[5]//env_test.num_row_choices*env_test.window), 't5')
            ims_test.append([im_target])



            for t_test in range(MAX_STEPS):
                action_test = select_action(state_test, env_test)
                
                state_test, reward_test, done_test, _ = env_test.step(action_test)

                ep_reward_test = reward_test + GAMMA*ep_reward_test
                im_test = plt.imshow(torch.from_numpy(state_test[1, :, :]), animated=True)
                ims_test.append([im_test])
                
                if done_test: 
                    break

        
            plt.title('epi_len_test = %i, reward_test = %f'%(t_test, ep_reward_test))
            ani_test = animation.ArtistAnimation(fig_test, ims_test, repeat=False, interval=500)#, interval=50000, blit=True)
            video_test_name = os.path.join(RESULT_DIR, 'test_%i.mp4'%(i_episode))
            ani_test.save(video_test_name)
            ######### end test video #######
            # episode_R_t = []
            # episode_len_t = []
            # episode_1st_action_Col_t = []
            # episode_1st_action_Row_t = [] 
            # for t_idx in range(TEST_NUM):
            #     env_t = img_env_brightest_patches_continuous.ImgEnv('mnist', train=False, max_steps=MAX_STEPS, channels=2, window=WINDOW, num_labels=NUM_LABELS)
            #     state_t, ep_reward_t = env_t.reset(), 0
            #     for t_t in range(MAX_STEPS):
            #         action_t = select_action(state_t, env_t)
                    
            #         state_t, reward_t, done_t, _ = env_t.step(action_t)

            #         ep_reward_t = reward_t + GAMMA*ep_reward_t
                    
            #         if done_t: 
            #             break


    plt.subplot(511)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(smoothing_average(episode_R))

    plt.subplot(512)
    plt.xlabel('Episode')
    plt.ylabel('Epi len')
    plt.plot(smoothing_average(episode_len))


    plt.subplot(513)
    plt.xlabel('Episode')
    plt.ylabel('1st Row')
    plt.plot(smoothing_average(episode_1st_action_Row))
    # plt.ylabel('policy_losses')
    # plt.plot(smoothing_average(episode_policy_losses))

    plt.subplot(514)
    plt.xlabel('Episode')
    plt.ylabel('1st Col')
    plt.plot(smoothing_average(episode_1st_action_Col))
    # plt.ylabel('value_losses')
    # plt.plot(smoothing_average(episode_value_losses))
    # plt.savefig(os.path.join(RESULT_DIR, 'AC_MNIST_brightest_patches_continuous_%ie'%(i_episode+1)))
    plt.subplot(515)
    plt.xlabel('Episode')
    plt.ylabel('tot loss')
    plt.plot(smoothing_average(episode_tot_loss))
    plt.show()

        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break

 
    























