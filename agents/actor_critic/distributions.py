import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import init, init_normc_, AddBias
import pdb

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical



old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)


FixedMultivariateNormal = torch.distributions.MultivariateNormal

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return x, FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))

        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return x, FixedNormal(action_mean, action_logstd.exp())



class TwoDGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(TwoDGaussian, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.mean1 = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std1 = init_(nn.Linear(num_inputs, num_outputs))
        # self.sigmoid1 = nn.Sigmoid()
        self.mean2 = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std2 = init_(nn.Linear(num_inputs, num_outputs))
        # self.sigmoid2 = nn.Sigmoid()
        
    def forward(self, x):
        action_mean = torch.cat((self.mean1(x)[0], self.mean2(x)[0]))
        # action_mean = torch.cat((self.sigmoid1(self.mean1(x))[0], (self.sigmoid2(self.mean2(x))[0])))
        action_mean = action_mean.to(device)
        action_std = torch.eye(2)
        log_std1 = torch.clamp(self.log_std1(x), -20, 2)
        log_std2 = torch.clamp(self.log_std2(x), -20, 2)
        
        action_std[0,0] = log_std1.exp()
        action_std[1,1] = log_std2.exp()
    
        action_std = action_std.to(device)
        return FixedMultivariateNormal(action_mean, action_std)
        
