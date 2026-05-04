import numpy as np
from agent_gfe import GFEAgent
from agent_bfe import BFEAgent
from agent_gfe_param import generate_gfe_params
from agent_bfe_param import generate_bfe_params
from environment import Environment
from environment_param import a, b, d

class World:
    def __init__(self, agent_type, time_horizon=3):
        self.time_horizon = time_horizon
        if agent_type == 'gfe':
            (A, B, C, D, U) = generate_gfe_params()
            self.agent = GFEAgent(A, B, C, D, U, time_horizon)
        elif agent_type == 'bfe':
            (A, B, C, D, U) = generate_bfe_params()
            self.agent = BFEAgent(A, B, C, D, U, time_horizon)
        self.environment = Environment(a, b, d)
        self.observations = np.zeros(self.time_horizon)

    def run(self):
        for k in range(self.time_horizon - 1):
            observation = self.environment.generate_observation()
            self.observations[k] = observation
            self.agent.observe(observation)
            self.agent.infer()
            action = self.agent.act()
            self.environment.act_upon(action)
        observation = self.environment.generate_observation()
        self.observations[self.time_horizon - 1] = observation

    def get_observations(self):
        return self.observations