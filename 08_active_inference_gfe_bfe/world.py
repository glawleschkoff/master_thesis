import numpy as np
from agent_gfe import Agent
from agent_gfe_param import A, B, C, D, U
from environment import Environment
from environment_param import a, b, d

class World:
    def __init__(self, time_horizon=3):
        self.time_horizon = time_horizon
        self.agent = Agent(A, B, C, D, U, time_horizon)
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

    def get_observations(self):
        observation = self.environment.generate_observation()
        self.observations[-1] = observation
        return self.observations