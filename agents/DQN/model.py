import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_

from resnet import resnet18
from context import MNIST_env
from MNIST_env.img_env_orig import IMG_ENVS


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, recurrent_policy, dataset=None, resnet=False, pretrained=False):
        super(Policy, self).__init__()
        self.dataset = dataset
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], recurrent_policy, dataset=dataset)
        elif len(obs_shape) == 1:
            assert not recurrent_policy, \
                "Recurrent policy is not implemented for the MLP controller"
            self.base = MLPBase(obs_shape[0])
        else:
            raise NotImplementedError
        if resnet:
            self.base = resnet18(channels=obs_shape[0], dataset=dataset, pretrained=pretrained)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        if dataset in ['mnist', 'cifar10']:
            self.clf = Categorical(self.base.output_size, 10)
        elif dataset in ['cifar100']:
            self.clf = Categorical(self.base.output_size, 100)
        self.state_size = self.base.state_size

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)
        self.actor_features = actor_features
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        if self.dataset in IMG_ENVS:
            clf = self.clf(self.actor_features)
            if deterministic:
                classif = clf.mode()
            else:
                classif = clf.sample()
            action = torch.cat([action, classif], 1)
            action_log_probs += clf.log_probs(classif)
        return value, action, action_log_probs, states


    def get_value(self, inputs, states, masks):
        value, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        if self.dataset in IMG_ENVS:
            clf = self.clf(actor_features)
            action_log_probs1 = clf.log_probs(action[:, 1])
            action = action[:, 0]
        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()
        if self.dataset in IMG_ENVS:
            action_log_probs += action_log_probs1
            dist_entropy += clf.entropy().mean()
        return value, action_log_probs, dist_entropy, states


class CNNBase(nn.Module):
    def __init__(self, num_inputs, use_gru, dataset=None):
        super(CNNBase, self).__init__()

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        if dataset == 'mnist':
            self.main = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 10, 5, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(10, 20, 5)),
                nn.ReLU(),
                init_(nn.Conv2d(20, 10, 5)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(360, 512)),
                nn.ReLU()
            )
        elif dataset in ['cifar10', 'cifar100']:
            self.main = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 6, kernel_size=5, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(6, 16, 5)),
                nn.ReLU(),
                init_(nn.Conv2d(16, 6, 5)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(216, 512)),
                nn.ReLU()
            )
        elif dataset == 'cityscapes':
            self.main = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 32, 8, stride=7)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(32 * 7 * 7, 512)),
                nn.ReLU()
            )
        else:
            self.main = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(32 * 7 * 7, 512)),
                nn.ReLU()
            )
        if use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.train()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def forward(self, inputs, states, masks):
        x = self.main(inputs)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
                N = states.size(0)
                T = int(x.size(0) / N)

                # unflatten
                x = x.view(T, N, x.size(1))

                # Same deal with masks
                masks = masks.view(T, N, 1)

                outputs = []
                for i in range(T):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)

                # assert len(outputs) == T
                # x is a (T, N, -1) tensor
                x = torch.stack(outputs, dim=0)
                # flatten
                x = x.view(T * N, -1)

        return x, states


class MLPBase(nn.Module):
    def __init__(self, num_inputs):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(64, 1))

        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 64

    def forward(self, inputs, states, masks):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor, states
