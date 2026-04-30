import numpy as np


class TMazeWorld:
    def __init__(self, model):
        self.A_true = model.A.copy()
        self.B_true_func = model.B
        self.D = model.D.copy()

    def reset(self):
        self.true_context = np.random.choice([0, 1])
        self.true_location = 0
        self.true_state_index = self._get_state_index()

        return self._get_observation()

    def _get_state_index(self):
        return self.true_location * 2 + self.true_context

    def _get_observation(self):
        p_obs_distribution = self.A_true[:, self.true_state_index]
        obs_index = np.random.choice(16, p=p_obs_distribution)
        o_t = np.zeros((16, 1))
        o_t[obs_index] = 1.0

        return o_t

    def step(self, action_u):
        B_matrix_true = self.B_true_func(action_u)
        p_next_state = B_matrix_true[:, self.true_state_index]
        self.true_state_index = np.random.choice(8, p=p_next_state)
        self.true_location = self.true_state_index // 2
        self.true_context = self.true_state_index % 2

        return self._get_observation()
