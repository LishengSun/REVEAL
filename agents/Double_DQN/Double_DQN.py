from agent_abstract import Agent_segment
from replay memory import ReplayMemory, Transition
from model_segment import navigation_model
from oracle_segment import oracle

import torch
import torch.optim as optim
from model_segment import device

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DDQN_separated_net(Agent_segment):
    def __init__(self, epsilon=0.3, memory_size=300, batch_size=16, model=navigation_model,
                 target_update_interval=1,
                 tau=0.005):
        super(DDQN_separated_net, self).__init__(epsilon=epsilon,
                                                 random_can_stop=False)

        # Memory
        self.memory = ReplayMemory(memory_size)

        # Batch size when learning
        self.batch_size = batch_size

        # number of time steps before an update of the delayed target Q network
        self.target_update_interval = target_update_interval

        # soft update weight of the delayed Q network
        self.tau = tau

    def learned_act(self, s, pred_oracle=True, online=False):
        if online:
            if pred_oracle:
                return torch.cat([self.model(s), oracle(s).unsqueeze(1)], 1)
        with torch.no_grad():
            if pred_oracle:
                return torch.cat([self.target_model(s), oracle(s).unsqueeze(1)], 1)
                # to do without oracle

    def reinforce(self, s_, a_, n_s_, r_, game_over_, env_steps_):
        # Two steps: first memorize the states, second learn from the pool

        self.memory.remember(s_, a_, n_s_, r_, game_over_)

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)


        # non_final_mask = torch.tensor(torch.cat(batch.game_over), device=device)==False
        non_final_mask = torch.cat(batch.game_over) == False

        non_final_next_states = torch.cat(batch.next_state)[non_final_mask]
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(-1, 2)
        reward_batch = torch.cat(batch.reward)
        # non_final_next_states = torch.cat(batch.next_state)[non_final_index]

        # print(state_batch.shape)
        state_values = self.learned_act(state_batch, online=True)
        state_action_values = torch.cat(
            [s[a[0].item(), a[1].item()].unsqueeze(0) for s, a in zip(state_values, batch.action)])

        next_state_values = torch.zeros(self.batch_size, device=device)

        if len(non_final_next_states) > 0:
            with torch.no_grad():
                argmax_online = (self.learned_act(non_final_next_states, online=True)).view(non_final_next_states.shape[0],-1).argmax(1)
                # print(torch.tensor(range(self.batch_size), device=device)[non_final_mask])
                # print(self.learned_act(non_final_next_states, online=False).view(-1, 2*SEGMENT_LENGTH).shape)
                next_state_values[non_final_mask] = \
                self.learned_act(non_final_next_states, online=False).view(non_final_next_states.shape[0], -1)[
                    range(len(non_final_next_states)), argmax_online]

        expected_state_action_values = next_state_values + reward_batch

        loss = F.smooth_l1_loss(state_action_values[non_final_mask],
                                expected_state_action_values[non_final_mask])  # .unsqueeze(1))
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

    def load_model(self, model_path='model.pickle', local=True):
        if local:
            self.model = navigation_model()
            self.target_model = navigation_model()
            hard_update(self.target_model, self.model)
        else:
            self.model = torch.load('model.pickle')
            self.target_model = torch.load('model.pickle')
        if torch.cuda.is_available():
            print('Using GPU')
            self.model.cuda()
            self.target_model.cuda()
        else:
            print('Using CPU')
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-5)