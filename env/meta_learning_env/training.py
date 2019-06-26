import numpy as np
import pickle
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(agent, env, epochs, epsilons, postfix='', draw=False, next_after_e=True, save_result=True,
          path_result="/content/drive/My Drive/oboe_RL/"):
    """
    Training function for the RL agent
    :param agent: (agent class) the RL agent. Careful : the agent should load a model (agent.load_model(*args)) before training it.
    :param env: (metalEnv class) the environment.
    :param epochs: (list of int) epochs where the epsilon should be decreased.
    :param epsilons:list of float) epsilon values.
    :param postfix: (string)
    :param draw: (bool)
    :param next_after_e: (bool) if the line should be changed after each trajectory
    :param save_result: (bool)
    :param path_result: (string)
    :return:
    """

    # Number of won games
    loss_list = []
    reward_list = []
    score = 0
    loss = 0
    for i, (epsilon, epoch) in enumerate(zip(epsilons, epochs)):
        agent.set_epsilon(epsilon)

        if i == 0:
            iterator = range(epoch)
        else:
            iterator = range(epochs[i-1], epoch)

        for e in iterator:
            # At each epoch, we restart to a fresh game and get the initial state
            state = env.reset(NEXT=next_after_e)
            state = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
            # This assumes that the games will terminate
            game_over = False

            tot_reward = 0
            tot_loss = 0
            while not game_over:
                # The agent performs an action
                action = agent.act(state)

                # Apply an action to the environment, get the next state, the reward
                # and if the games end
                prev_state = state
                state, reward, game_over = env.step(action.item())
                tot_reward += reward

                action = action.unsqueeze(0).unsqueeze(0)
                state = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
                reward = torch.tensor(reward, device=device, dtype=torch.float).unsqueeze(0)
                game_over_tensor = torch.tensor(game_over, device=device).unsqueeze(0)

                # Apply the reinforcement strategy
                loss = agent.reinforce(prev_state, action, state, reward, game_over_tensor, env.num_steps)
                tot_loss += loss

                #print((e, epoch, loss, tot_reward))

                # Save action
                if draw:
                    env.draw(postfix+str(e))

            # Update stats
            score += tot_reward
            loss_list.append(tot_loss/env.num_steps)
            reward_list.append(tot_reward)

            if e%50 == 0:
                print("Epoch {:03d}/{:03d} | Loss {:.4f} | reward {}".format(e, epoch, loss, tot_reward))


    if save_result:
        with open(path_result + "rewards_{}.pickle".format(postfix), 'wb') as handle:
            pickle.dump(reward_list, handle)

        with open(path_result + "losses_{}.pickle".format(postfix), 'wb') as handle:
            pickle.dump(loss_list, handle)

        agent.save_model(path_result + 'model_{}.pickle'.format(postfix))


def test(agent, env, epoch, postfix='', draw=False, path_result='/content/', save_result=True,
         rep_allowed=False):
    """
    :param agent: (agent class) the RL agent.
    :param env: (metalEnv class) the environment.
    :param epoch: (int) number of epochs.
    :param postfix: (string)
    :param draw: (bool)
    :param path_result: (string)
    :param save_result: (bool)
    :param rep_allowed: (bool) if the agent should try a new model at each iteration.
    :return: (list of int) history of lines, (list of list of int) history of trajectories
    """
    # Number of won games
    action_histories = []
    reward_list = []
    lines_list = []
    score = 0

    for e in range(epoch):
        # At each epoch, we restart to a fresh game and get the initial state
        state = env.reset(NEXT=True)
        state = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
        # This assumes that the games will terminate
        game_over = False

        tot_reward = 0

        while not game_over:
            # The agent performs an action
            if rep_allowed:
                action = agent.act(state, train=False).item()
            else:
                with torch.no_grad():
                    Q_values = agent.model(state)[0].cpu().numpy()
                    best_models = np.argsort(Q_values)
                    best_models_no_rep = [model for model in best_models if model not in env.action_history]
                    action = best_models_no_rep[-1]


            # Apply an action to the environment, get the next state, the reward
            # and if the games end
            prev_state = state
            state, reward, game_over = env.step(action)

            # update metrics
            tot_reward += reward

            state = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)

            # Save action
            if draw:
                env.draw(postfix + str(e))

        # Update stats
        score += tot_reward
        reward_list.append(tot_reward)
        action_histories.append(env.action_history)
        lines_list.append(env.line)

        if e % 50 == 0:
            print("Epoch {:03d}/{:03d} | reward {}".format(e, epoch, tot_reward))

    if save_result:
        with open(path_result + "rewards_test_{}.pickle".format(postfix), 'wb') as handle:
            pickle.dump(reward_list, handle)
    return lines_list, action_histories