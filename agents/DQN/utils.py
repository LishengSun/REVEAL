import torch
import torch.nn as nn
from context import MNIST_env
from MNIST_env.img_env import IMG_ENVS


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def update_current_obs(obs, current_obs, obs_shape, num_stack):
    shape_dim0 = obs_shape[0]
    obs = torch.from_numpy(obs).float()
    if num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs


def eval_episode(env, agent, args):
    agent.base.eval()
    obs = env.reset()
    obs_shape = env.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    current_obs = torch.zeros(1, *obs_shape)
    done = False
    total_reward = 0
    states = torch.zeros(1, 512)
    if args.cuda:
        current_obs = current_obs.cuda()
        states = states.cuda()
        agent.base.cuda()
    while not done:
        # state = torch.from_numpy(obs).float().cuda()
        update_current_obs(obs, current_obs, obs_shape, args.num_stack)
        value, action, action_log_probs, states = agent.act(
             current_obs, states, FloatTensor([[0.0]]),
             deterministic=True)
        obs, reward, done, _ = env.step(action.detach().cpu().numpy())
        total_reward += reward[0]
    return total_reward
