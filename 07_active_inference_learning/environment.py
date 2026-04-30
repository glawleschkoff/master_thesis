import numpy as np

class Environment:
    def __init__(self, a, b, d):
        self.a = a
        self.b = b
        self.d = d
        self.state = None

    def reset(self):
        self.state = None

    def generate_observation(self):
        if self.state == None:
            number_states = len(self.d)
            self.state = np.random.choice(number_states, p=self.d)
        p_o = self.a[:, self.state]
        number_observations = len(p_o)
        observation = np.random.choice(number_observations, p=p_o)
        return observation
    
    def act_upon(self, action):
        if self.state == None:
            number_states = len(self.d)
            self.state = np.random.choice(number_states, p=self.d)
        p_s = self.b[:, self.state, action]
        number_states = len(p_s)
        state = np.random.choice(number_states, p=p_s)
        self.state = state

