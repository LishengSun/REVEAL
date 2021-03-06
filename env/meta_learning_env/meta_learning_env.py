import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import seaborn as sns
import os


class metalEnv(object):
    def __init__(self, loss_matrix=pd.read_csv(os.path.join(os.getcwd(), "meta_learning_matrices/error_BER_withoutNa.csv")),
                 time_matrix=pd.read_csv(os.path.join(os.getcwd(), "meta_learning_matrices/Time_withoutNa.csv")),
                 metafeatures_matrix=pd.read_csv(os.path.join(os.getcwd(), "meta_learning_matrices/Meta_features_withoutNa.csv")),
                 compression=None, noise=0.0, time_cost=0.1, max_steps=20, use_meta_features=False, use_case=None):
        self.compression = compression
        self.noise = noise

        self.nb_models = time_matrix.shape[1] - 1
        self.nb_metafeatures = metafeatures_matrix.shape[1] - 1
        self.segment_length = self.nb_models

        self.time_matrix = time_matrix.iloc[:, 1:] # keep only real data

        # meta features
        self.use_meta_features = use_meta_features
        if use_meta_features:
            metafeatures_matrix.iloc[:, 1:] = (metafeatures_matrix.iloc[:, 1:] -
                                        metafeatures_matrix.iloc[:, 1:].mean()) / metafeatures_matrix.iloc[:, 1:].std()
            loss_matrix = loss_matrix.merge(metafeatures_matrix, on="dataset")
            self.metafeatures_matrix = loss_matrix.iloc[:, (self.nb_models + 1):]
            self.segment_length += self.nb_metafeatures

        # keep only real data
        self.loss_matrix = loss_matrix.iloc[:, 1:(self.nb_models+1)]

        self.max_steps = max_steps
        self.time_cost = time_cost

        # these attributes are here to make plotting easier
        self.to_draw = np.zeros((self.max_steps + 1, 1, self.segment_length, 3)).astype(int)
        self.use_case = use_case


        if self.use_case == 'train':
            self.loss_matrix = self.loss_matrix.iloc[:int(0.7 * self.loss_matrix.shape[0]), :]
            self.time_matrix = self.time_matrix.iloc[:int(0.7 * self.time_matrix.shape[0]), :]
            if use_meta_features:
                self.metafeatures_matrix = self.metafeatures_matrix.iloc[:int(0.7 * self.metafeatures_matrix.shape[0]), :]
        elif self.use_case == 'test':
            self.loss_matrix = self.loss_matrix.iloc[int(0.7 * self.loss_matrix.shape[0]):, :]
            self.time_matrix = self.time_matrix.iloc[int(0.7 * self.time_matrix.shape[0]):, :]
            if use_meta_features:
                self.metafeatures_matrix = self.metafeatures_matrix.iloc[int(0.7 * self.metafeatures_matrix.shape[0]):, :]
        else:
            self.loss_matrix = self.loss_matrix
            self.time_matrix = self.time_matrix
            if use_meta_features:
                self.metafeatures_matrix = self.metafeatures_matrix

        # self.loss_matrix.reset_index(drop=True, inplace=True)
        # self.time_matrix.reset_index(drop=True, inplace=True)
        # if use_meta_features:
        #    self.metafeatures_matrix.reset_index(drop=True, inplace=True)

    def seed(self, seed):
        np.random.seed(seed)

    def generate_a_segment(self, line=None):
        if line is None:
            line = random.randint(self.loss_matrix.index[0], self.loss_matrix.index[-1]-1)

        noise = self.noise
        segment = self.loss_matrix.loc[line]
        if self.use_meta_features:
            segment = np.concatenate((segment, self.metafeatures_matrix.loc[line]))
        return line, segment + noise * np.random.random(self.segment_length)

    def reset(self, NEXT=True, line=None):
        self.pos = None
        self.num_steps = 0
        self.total_time = 0
        self.action_history = []
        if NEXT:
            number, line = self.generate_a_segment(line)
            self.line_number = number
            self.current_line = line

        self.state = np.zeros((2, self.segment_length))

        if self.use_meta_features:
            self.state[1, -self.nb_metafeatures:] = self.metafeatures_matrix.loc[self.line_number]
            self.state[0, -self.nb_metafeatures:] = np.ones(self.nb_metafeatures)
        return self.state

    def step(self, action):
        """
        Compute 1 step of the game.
        :param action: int in range(self.time_matrix.shape[1])
        :return: array new_state, float reward, bool done
        """
        if self.pos is None:
            reward = -self.time_cost * self.time_matrix.loc[self.line_number][action] + 1 - self.loss_matrix.loc[
                self.line_number][action]
        else:
            reward = -self.time_cost * self.time_matrix.loc[self.line_number][action] + max(0, np.min(
                self.loss_matrix.loc[self.line_number][self.action_history]) - self.loss_matrix.loc[self.line_number][action])
        self.pos = action

        self.state[1, self.pos] = self.current_line[self.pos]
        self.state[0, self.pos] = 1.

        self.action_history.append(action)
        self.get_frame(int(self.num_steps))
        self.num_steps += 1
        done = self.num_steps == self.max_steps
        return self.state, reward, done

    def get_current_obs(self):
        return self.state

    def close(self):
        pass

    # The following functions have to do with plotting performance through time. At the end of a trajectory, simply
    # call plot_time_perf
    def _better_than_random_baseline(self, line):
        """
        compute a simple greedy baseline that tries alogrithms from the fastest to the slowest

        :param line: the number of the line fo the matrix to consider
        :return: lists of the errors, times and names of the choosen algorithms
        """
        possible_index = np.argsort(self.time_matrix.loc[line])[:200]
        chosen_index = possible_index
        error = self.loss_matrix.loc[line][chosen_index]
        time = self.time_matrix.loc[line][chosen_index]
        algo = self.loss_matrix.columns[chosen_index]
        return error, time, algo

    def plot_time_perf(self, line=None, action_history=None):

        """
        At the end of the trajectory, plots the performance-time graph.

        :param line: line number for a specific dataset
        :param action_history: (list of int)
        :return:
        """
        if not line:
            line = self.line_number
        if not action_history:
            action_history = self.action_history
        algo = self.loss_matrix.columns[action_history]
        error = self.loss_matrix.iloc[line, action_history]
        time = self.time_matrix.iloc[line, action_history]

        def _minimums_index(error, time):
            return [0] + [i for i in range(1, len(time)) if error[i] < np.min(error[:i])]

        sns.lineplot(np.cumsum(time), np.minimum.accumulate(error))
        plt.vlines(np.cumsum(time), 0, max(np.minimum.accumulate(error)), linestyles='dashed')
        plt.xlabel("Time (s)")
        plt.ylabel("BER")
        plt.show()

        optimal = np.min(self.loss_matrix.loc[line])
        cumtime = np.cumsum(time)
        min_ind = _minimums_index(error, time)
        cum_error = np.minimum.accumulate(error)

        error_random, time_random, algo_random = self._better_than_random_baseline(line)
        cumtime_random = np.cumsum(time_random)[np.cumsum(time_random) < np.max(cumtime)]
        cum_error_random = np.minimum.accumulate(error_random)[np.cumsum(time_random) < np.max(cumtime)]

        sns.lineplot(cumtime, cum_error, label='agent')
        sns.lineplot(cumtime_random, cum_error_random, label='greedy baseline')
        plt.vlines(cumtime[min_ind], 0, max(max(cum_error), max(cum_error_random)), linestyles='dashed')
        plt.hlines(optimal, min(cumtime), max(cumtime), label='optimal model', color='r')
        for i in min_ind:
            plt.text(cumtime[i], max(max(cum_error), max(cum_error_random)) / 1.2, algo[i][15:].split(" ")[0][:-2],
                     rotation=90, color='k')
        plt.xlabel("Time (s)")
        plt.ylabel("BER")
        plt.legend()
        plt.show()

    def plot_time_perf_mean(self, lines, action_histories):

        def _minimums_index(error, time):
            return [0] + [i for i in range(1, len(time)) if error[i] < np.min(error[:i])]

        time_total = []
        line_total = []
        for (line, action_history) in zip(lines, action_histories):
            time = self.time_matrix.iloc[line, action_history]
            time_total = time_total + time
            line_total = line_total + [line]*len(time)
        ind_sort = np.argsort(time_total)
        line_total = np.array(line_total)[ind_sort]
        time_total = np.array(time_total)[ind_sort]
        cum_error_total

    # the following functions have to do with plotting the trajectory as lighting pixels in the segment
    # (as in the segment game)
    def get_frame(self, t):
        segment_plot = np.zeros((1, self.segment_length, 3)).astype(int)
        segment_plot[:, self.state[0] == 0, :] = segment_plot[:, self.state[0] == 0, :] + 128
        segment_plot[:, self.state[0] == 1, 0] = (self.current_line[self.state[0] == 1] * 255).astype(
            int)
        if self.pos is not None:
            segment_plot[:, self.pos, :] = np.clip(segment_plot[:, self.pos, :] + 170, 0, 255)

        self.to_draw[t, :, :, :] = segment_plot

    def draw(self, e):
        true_image = np.zeros((1, self.segment_length, 3)).astype(int)
        true_image[:, :, 0] = (self.current_line * 255).astype(int)

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

