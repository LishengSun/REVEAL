import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent_abstract import Agent
from replay memory import ReplayMemory, Transition

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
GAMMA = 0.99

# Q value network
class value_network_full_segment(nn.Module):
  def __init__(self, segment_length):
    super(value_network_full_segment, self).__init__()
    self.conv1 = nn.Conv1d(2, 1, kernel_size=1, padding=0)
    self.linear1 = nn.Linear(220, 60)
    self.linear2 = nn.Linear(60, 60)
    self.linear3 = nn.Linear(60, 220)
  def forward(self, x):
    x = F.relu(self.conv1(x))
    #print(x.shape)
    x = F.relu(self.linear1(x))
    #print(x.shape)
    x = F.relu(self.linear2(x))
    #print(x.shape)
    x = self.linear3(x)
    #print(x.shape)
    x = x.view(x.size(0), -1)
    return x

# target network updates for the DDQN
def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
      target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
  for target_param, param in zip(target.parameters(), source.parameters()):
      target_param.data.copy_(param.data)


class DDQN_separated_net(Agent):
    def __init__(self, segment_length=220, epsilon=0.3, memory_size=300, batch_size=16, target_update_interval=1,
                 tau=0.005):
        super(DDQN_separated_net, self).__init__(epsilon=epsilon, segment_length=segment_length)

        # Memory
        self.memory = ReplayMemory(memory_size)

        # Batch size when learning
        self.batch_size = batch_size

        #
        self.target_update_interval = target_update_interval

        #
        self.tau = tau

    def learned_act(self, s, with_grad=False, target=False):
        if with_grad:
            return self.model(s)
        with torch.no_grad():
            if target:
                return self.target_model(s)
            return self.model(s)
            # to do without oracle

    def reinforce(self, s_, a_, n_s_, r_, game_over_, env_steps_):
        # Two steps: first memorize the states, second learn from the pool

        self.memory.remember(s_, a_, n_s_, r_, game_over_)

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # print(batch.state)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        # non_final_mask = torch.tensor(torch.cat(batch.game_over), device=device)==False
        non_final_mask = torch.cat(batch.game_over) == False

        non_final_next_states = torch.cat(batch.next_state)[non_final_mask]
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # non_final_next_states = torch.cat(batch.next_state)[non_final_index]

        # print(state_batch.shape)
        state_values = self.learned_act(state_batch, with_grad=True)
        state_action_values = state_values.gather(1, action_batch).squeeze(1)

        next_state_values = torch.zeros(self.batch_size, device=device)

        if len(non_final_next_states) > 0:
            with torch.no_grad():
                argmax_online = (self.learned_act(non_final_next_states)).argmax(1).unsqueeze(1)
                next_state_values[non_final_mask] = self.learned_act(non_final_next_states,
                                                                     target=True).gather(1,argmax_online).squeeze(1)

        expected_state_action_values = next_state_values * GAMMA + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        # loss = F.mse_loss(state_action_values[non_final_mask], expected_state_action_values[non_final_mask])

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            # HINT: Clip the target to avoid exploiding gradients.. -- clipping is a bit tighter
            param.grad.data.clamp_(-1e-6, 1e-6)
        self.optimizer.step()

        if env_steps_ % self.target_update_interval == 0:
            soft_update(self.target_model, self.model, self.tau)

        return float(loss)

    def save_model(self, model_path='model.pickle'):
        try:
            torch.save(self.model, model_path)
        except:
            pass

    def load_model(self, model=value_network_full_segment, model_path='model.pickle', local=True):
        if local:
            # self.model = big_navigation_model()
            # self.target_model = big_navigation_model()
            self.model = model(segment_length=self.segment_length)
            self.target_model = model(segment_length=self.segment_length)
            hard_update(self.target_model, self.model)
        else:
            self.model = torch.load(model_path)
            self.target_model = torch.load(model_path)
        if torch.cuda.is_available():
            print('Using GPU')
            self.model.cuda()
            self.target_model.cuda()
        else:
            print('Using CPU')
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-5)