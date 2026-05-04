import numpy as np
from utils import safelog
np.set_printoptions(
    linewidth=200,       # Längere Zeilen
    precision=3,         # Nur 3 Nachkommastellen anzeigen
    suppress=True        # Verhindert die wissenschaftliche Notation (z.B. 1e-05 wird zu 0.000)
)

class BFEAgent:
    def __init__(self, A, B, C, D, U, time_horizon):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.U = U
        self.time_horizon = time_horizon

        self.current_observation_timestep = 0
        self.current_action_timestep = 0

        self.number_states = B.shape[0]
        self.number_observations = A.shape[0]
        self.number_actions = B.shape[2]
        self.state_beliefs = np.ones((self.time_horizon, self.number_states)) * (1 / self.number_states)
        self.observation_beliefs = np.ones((self.time_horizon, self.number_observations)) * (1 / self.number_observations)
        self.action_beliefs = np.ones((self.time_horizon - 1, self.number_actions)) * (1 / self.number_actions)
        self.A_beliefs = np.ones((self.time_horizon, self.number_observations, self.number_states))  * (1 / self.number_observations)
        self.B_beliefs = np.ones((self.time_horizon - 1, self.number_states, self.number_states, self.number_actions)) * (1 / self.number_states)


    def free_energy(self):
        # energies
        D_energy = np.array([-np.einsum('i,i', self.state_beliefs[0], safelog(self.D))])
        A_energies = np.zeros(self.time_horizon)
        B_energies = np.zeros(self.time_horizon - 1)
        C_energies = np.zeros(self.time_horizon)
        U_energies = np.zeros(self.time_horizon - 1)
        for k in range(self.time_horizon):
            A_energies[k] = -np.einsum('ij,ij', self.A_beliefs[k], safelog(self.A))
            C_energies[k] = -np.einsum('i,i', self.observation_beliefs[k], safelog(self.C))
        for k in range(self.time_horizon - 1):
            B_energies[k] = -np.einsum('ijk,ijk', self.B_beliefs[k], safelog(self.B))
            U_energies[k] = -np.einsum('i,i', self.action_beliefs[k], safelog(self.U))

        # entropies
        state_entropies = np.zeros(self.time_horizon)
        observation_entropies = np.zeros(self.time_horizon)
        action_entropies = np.zeros(self.time_horizon - 1)
        A_entropies = np.zeros(self.time_horizon)
        B_entropies = np.zeros(self.time_horizon - 1)
        for k in range(self.time_horizon):
            state_entropies[k] = -np.einsum('i,i', self.state_beliefs[k], safelog(self.state_beliefs[k]))
            observation_entropies[k] = -np.einsum('i,i', self.observation_beliefs[k], safelog(self.observation_beliefs[k]))
            A_entropies[k] = -np.einsum('ij,ij', self.A_beliefs[k], safelog(self.A_beliefs[k]))
        for k in range(self.time_horizon - 1):
            action_entropies[k] = -np.einsum('i,i', self.action_beliefs[k], safelog(self.action_beliefs[k]))
            B_entropies[k] = -np.einsum('ijk,ijk', self.B_beliefs[k], safelog(self.B_beliefs[k]))

        # free energies
        D_free_energy = D_energy - state_entropies[0]
        A_free_energies = np.zeros(self.time_horizon)
        B_free_energies = np.zeros(self.time_horizon - 1)
        C_free_energies = np.zeros(self.time_horizon)
        U_free_energies = np.zeros(self.time_horizon - 1)
        for k in range(self.time_horizon):
            A_free_energies[k] = A_energies[k] - A_entropies[k]
            C_free_energies[k] = C_energies[k] - observation_entropies[k]
        for k in range(self.time_horizon - 1):
            B_free_energies[k] = B_energies[k] - B_entropies[k]
            U_free_energies[k] = U_energies[k] - action_entropies[k]

        # total energy
        total_energy = np.zeros(1)
        total_energy += np.sum(D_energy)
        total_energy += np.sum(A_energies)
        total_energy += np.sum(B_energies)
        total_energy += np.sum(U_energies)
        total_energy += np.sum(C_energies)
        

        # total entropy
        total_entropy = np.zeros(1)
        total_entropy += np.sum(A_entropies)
        total_entropy += np.sum(B_entropies)
        total_entropy -= state_entropies[0]
        total_entropy -= 2*state_entropies[1]
        total_entropy -= state_entropies[2]

        total_free_energy = total_energy - total_entropy

        return (total_free_energy, total_energy, total_entropy, A_free_energies, B_free_energies, C_free_energies, D_free_energy, U_free_energies,
                A_energies, B_energies, C_energies, D_energy, U_energies,
                A_entropies, B_entropies, state_entropies, observation_entropies, action_entropies)


    def observe(self, observation):
        self.observation_beliefs[self.current_observation_timestep, :] = 0
        self.observation_beliefs[self.current_observation_timestep, observation] = 1
        self.current_observation_timestep += 1


    def infer(self):
        tolerance = 1e-4
        previous_free_energy = 0
        current_free_energy = float('inf')

        while (abs(current_free_energy - previous_free_energy) > tolerance):
        #for _ in range(10):
            previous_free_energy = current_free_energy

            # messages for forward sweep
            rightward_state_messages = np.zeros((self.time_horizon, self.number_states))
            upward_state_messages = np.zeros((self.time_horizon, self.number_states))
            downward_action_messages = np.zeros((self.time_horizon - 1, self.number_actions))
            upward_observation_messages = np.zeros((self.time_horizon, self.number_observations))

            # messages for backward sweep
            upward_action_messages = np.zeros((self.time_horizon - 1, self.number_actions))
            leftward_state_messages = np.zeros((self.time_horizon - 1, self.number_states))
            downward_observation_messages = np.zeros((self.time_horizon, self.number_observations))

            # forward sweep (0 to T-1)
            for k in range(self.time_horizon):
                upward_observation_messages[k] = self.C
                if k == 0:
                    rightward_state_messages[0] = self.D
                else:
                    rightward_state_messages[k] = np.einsum('ijk,j,j,k->i', self.B, rightward_state_messages[k - 1], upward_state_messages[k - 1], downward_action_messages[k - 1])
                if self.current_observation_timestep > k:
                    upward_state_messages[k] = self.A[np.argmax(self.observation_beliefs[k])]
                else:
                    upward_state_messages[k] = np.exp(np.einsum('ij,i->j', safelog(self.A), self.C))
                if k < self.time_horizon - 1:
                    if k < self.current_action_timestep:
                        downward_action_messages[k] = self.U
                    else:
                        downward_action_messages[k] = self.U

            # backward sweep (T-1 to 0)
            for k in reversed(range(self.time_horizon)):
                if k == self.time_horizon - 1:
                    downward_observation_messages[k] = np.exp(np.einsum('ij,j->i', safelog(self.A), rightward_state_messages[k]))
                elif k == self.time_horizon - 2:
                    upward_action_messages[k] = np.einsum('ijk,j,j,i->k', self.B, rightward_state_messages[k], upward_state_messages[k], upward_state_messages[k + 1])
                    leftward_state_messages[k] = np.einsum('ijk,k,i->j', self.B, downward_action_messages[k], upward_state_messages[k + 1])
                    downward_observation_messages[k] = np.exp(np.einsum('ij,j,j->i', safelog(self.A), rightward_state_messages[k], leftward_state_messages[k]))
                else:
                    upward_action_messages[k] = np.einsum('ijk,j,j,i,i->k', self.B, rightward_state_messages[k], upward_state_messages[k], upward_state_messages[k + 1], leftward_state_messages[k + 1])
                    leftward_state_messages[k] = np.einsum('ijk,k,i,i->j', self.B, downward_action_messages[k], upward_state_messages[k + 1], leftward_state_messages[k + 1])
                    downward_observation_messages[k] = np.exp(np.einsum('ij,j,j->i', safelog(self.A), rightward_state_messages[k], leftward_state_messages[k]))
                
            # belief updating
            for k in range(self.time_horizon):
                if k == self.time_horizon - 1:
                    self.state_beliefs[k] = (rightward_state_messages[k] * upward_state_messages[k]) / np.sum(rightward_state_messages[k] * upward_state_messages[k])
                else:  
                    self.state_beliefs[k] = (rightward_state_messages[k] * upward_state_messages[k] * leftward_state_messages[k]) / np.sum(rightward_state_messages[k] * upward_state_messages[k] * leftward_state_messages[k]) 
            for k in range(self.current_observation_timestep + 1, self.time_horizon):
                self.observation_beliefs[k] = (upward_observation_messages[k] * downward_observation_messages[k]) / np.sum(upward_observation_messages[k] * downward_observation_messages[k])
            for k in range(self.time_horizon - 1):
                self.action_beliefs[k] = (downward_action_messages[k] * upward_action_messages[k]) / np.sum(downward_action_messages[k] * upward_action_messages[k])

            result = self.free_energy()
            current_free_energy = result[0]


    def act(self, action = None):
        if action == None:
            current_action_belief = self.action_beliefs[self.current_action_timestep]
            number_actions = len(current_action_belief)
            action = np.random.choice(number_actions, p=current_action_belief)
        self.current_action_timestep += 1
        return action
    

    def print_beliefs(self):
        print('Beliefs (t=0):')
        print('q(s_0) = ' + str(self.state_beliefs[0]))
        print('q(o_0) = ' + str(self.observation_beliefs[0]))
        print('q(u_0) = ' + str(self.action_beliefs[0])) 
        print()

        print('Beliefs (t=1):')
        print('q(s_1) = ' + str(self.state_beliefs[1]))
        print('q(o_1) = ' + str(self.observation_beliefs[1]))
        print('q(u_1) = ' + str(self.action_beliefs[1])) 
        print()

        print('Beliefs (t=2):')
        print('q(s_2) = ' + str(self.state_beliefs[2]))
        print('q(o_2) = ' + str(self.observation_beliefs[2]))
        print()