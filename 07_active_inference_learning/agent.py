import numpy as np
import scipy.special as sp
from utils import optimize_q_s, optimize_q_s_iterative, safelog
np.set_printoptions(
    linewidth=200,       # Längere Zeilen
    precision=3,         # Nur 3 Nachkommastellen anzeigen
    suppress=True        # Verhindert die wissenschaftliche Notation (z.B. 1e-05 wird zu 0.000)
)

class Agent:
    def __init__(self, alpha, B, C, D, U, time_horizon):
        self.alpha = alpha
        self.B = B
        self.C = C
        self.D = D
        self.U = U
        self.time_horizon = time_horizon

        self.current_observation_timestep = 0
        self.current_action_timestep = 0

        self.number_states = B.shape[0]
        self.number_observations = alpha.shape[0]
        self.number_actions = B.shape[2]
        self.state_beliefs = np.ones((self.time_horizon, self.number_states)) * (1 / self.number_states)
        self.observation_beliefs = np.ones((self.time_horizon, self.number_observations)) * (1 / self.number_observations)
        self.action_beliefs = np.ones((self.time_horizon - 1, self.number_actions)) * (1 / self.number_actions)

    def reset(self):
        self.current_observation_timestep = 0
        self.current_action_timestep = 0
        self.state_beliefs = np.ones((self.time_horizon, self.number_states)) * (1 / self.number_states)
        self.observation_beliefs = np.ones((self.time_horizon, self.number_observations)) * (1 / self.number_observations)
        self.action_beliefs = np.ones((self.time_horizon - 1, self.number_actions)) * (1 / self.number_actions)

    def learn_observation_likelihood(self):
        for k in range(1, self.time_horizon):
            observation = np.argmax(self.observation_beliefs[k])
            self.alpha[observation, :] += self.state_beliefs[k]
    
    def free_energy(self, rightward_state_messages, upward_state_messages, downward_action_messages, upward_action_messages, leftward_state_messages):
        # joint beliefs
        A_beliefs = np.zeros((self.time_horizon, self.number_observations, self.number_states))
        B_beliefs = np.zeros((self.time_horizon - 1, self.number_states, self.number_states, self.number_actions))
        for k in range(self.time_horizon):
            A_beliefs[k, :, :] = np.einsum('i,j->ij', self.observation_beliefs[k], self.state_beliefs[k])
        for k in range(self.time_horizon - 1):
            if k == self.time_horizon - 2:
                belief = np.einsum('ijk,j,j,k,i->ijk', self.B, rightward_state_messages[k], upward_state_messages[k], downward_action_messages[k], upward_state_messages[k + 1])
                B_beliefs[k, :, :, :] = belief / np.sum(belief)
            else:
                belief = np.einsum('ijk,j,j,k,i,i->ijk', self.B, rightward_state_messages[k], upward_state_messages[k], downward_action_messages[k], upward_state_messages[k + 1], leftward_state_messages[k + 1])
                B_beliefs[k, :, :, :] = belief / np.sum(belief)
        
        # energies
        D_energy = np.array([-np.einsum('i,i', self.state_beliefs[0], safelog(self.D))])
        A_energies = np.zeros(self.time_horizon)
        B_energies = np.zeros(self.time_horizon - 1)
        C_energies = np.zeros(self.time_horizon)
        U_energies = np.zeros(self.time_horizon - 1)
        for k in range(self.time_horizon):
            A_energies[k] = -np.einsum('ij,ij', A_beliefs[k], safelog(self.A))
            C_energies[k] = -np.einsum('i,i', self.observation_beliefs[k], safelog(self.C))
        for k in range(self.time_horizon - 1):
            B_energies[k] = -np.einsum('ijk,ijk', B_beliefs[k], safelog(self.B))
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
            A_entropies[k] = -np.einsum('ij,ij', A_beliefs[k], safelog(A_beliefs[k]))
        for k in range(self.time_horizon - 1):
            action_entropies[k] = -np.einsum('i,i', self.action_beliefs[k], safelog(self.action_beliefs[k]))
            B_entropies[k] = -np.einsum('ijk,ijk', B_beliefs[k], safelog(B_beliefs[k]))

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

        # total free energy
        total_free_energy = np.zeros(1)
        total_free_energy += np.sum(A_free_energies)
        total_free_energy += np.sum(B_free_energies)
        total_free_energy += np.sum(C_free_energies)
        total_free_energy += np.sum(D_free_energy)
        total_free_energy += np.sum(U_free_energies)
        total_free_energy -= np.sum(action_entropies)
        total_free_energy -= np.sum(state_entropies)
        total_free_energy -= np.sum(state_entropies[:-1])
        total_free_energy -= np.sum(observation_entropies)

        print('A_energies: ' + str(A_energies))
        print('B_energies: ' + str(B_energies))
        print('C_energies: ' + str(C_energies))
        print('D_energy: ' + str(D_energy))
        print('U_energies: ' + str(U_energies))
        print()
        print('A_entropies: ' + str(A_entropies))
        print('B_entropies: ' + str(B_entropies))
        print('state_entropies: ' + str(state_entropies))
        print('observation_entropies: ' + str(observation_entropies))
        print('action_entropies: ' + str(action_entropies))
        print()
        print('A_free_energies: ' + str(A_free_energies))
        print('B_free_energies: ' + str(B_free_energies))
        print('C_free_energies: ' + str(C_free_energies))
        print('D_free_energy: ' + str(D_free_energy))
        print('U_free_energies: ' + str(U_free_energies))
        print()
        print('total_free_energy: ' + str(total_free_energy))
        print()
        print('-' * 40)
        print()

        return total_free_energy


    def observe(self, observation):
        self.observation_beliefs[self.current_observation_timestep, :] = 0
        self.observation_beliefs[self.current_observation_timestep, observation] = 1
        self.current_observation_timestep += 1


    def evaluate_policies(self):
        remaining_actions = (self.time_horizon - 1) - self.current_action_timestep
        policies = np.indices((self.number_actions,) * remaining_actions).reshape(remaining_actions, -1).T
        policies_fe = np.ones(remaining_actions * self.number_actions) * np.finfo(np.float64).max

        for n, pi in enumerate(policies):
            policy_length = pi.size
            for i in range(policy_length):
                self.action_beliefs[self.current_action_timestep + i] = np.eye(self.number_actions)[pi[i]]
            #policies_fe[n] = infer()
            self.infer()
            self.print_beliefs()


    def infer(self):
        tolerance = 1e-4
        previous_free_energy = 0
        current_free_energy = float('inf')

        #while (abs(current_free_energy - previous_free_energy) > tolerance):
        for _ in range(30):
            previous_free_energy = current_free_energy

            # messages for forward sweep
            rightward_state_messages = np.zeros((self.time_horizon, self.number_states))
            upward_state_messages = np.zeros((self.time_horizon, self.number_states))
            downward_action_messages = np.zeros((self.time_horizon - 1, self.number_actions))

            # messages for backward sweep
            upward_action_messages = np.zeros((self.time_horizon - 1, self.number_actions))
            leftward_state_messages =  np.zeros((self.time_horizon - 1, self.number_states))

            # message forward sweep (0 to T-1)
            for k in range(self.time_horizon):
                if k == 0:
                    rightward_state_messages[0] = self.D
                else:
                    rightward_state_messages[k] = np.einsum('ijk,j,j,k->i', self.B, rightward_state_messages[k - 1], upward_state_messages[k - 1], downward_action_messages[k - 1])
                if self.current_observation_timestep > k:
                    #upward_state_messages[k] = self.A[np.argmax(self.observation_beliefs[k])]
                    upward_state_messages[k] = np.exp(sp.digamma(self.alpha[0,:]) - sp.digamma(np.sum(self.alpha, axis=0)))
                else:
                    #upward_state_messages[k] = np.exp(np.einsum('ij,ij->j', self.A, safelog((self.A * self.C[:, np.newaxis]) / np.clip(self.observation_beliefs[k][:, np.newaxis], a_min=np.finfo(float).eps, a_max=None))))
                    summand_1 = np.sum(self.alpha / np.sum(self.alpha, axis=0) * (sp.digamma(self.alpha + 1) - sp.digamma(np.sum(self.alpha, axis=0) + 1)), axis=0)
                    #summand_2 = np.sum(self.alpha / np.sum(self.alpha, axis=0) * (safelog(self.C) - safelog(self.observation_beliefs[k])), axis=0)
                    summand_2 = np.sum(np.einsum('ij,i->ij', self.alpha / np.sum(self.alpha, axis=0), safelog(self.C) - safelog(self.observation_beliefs[k])), axis=0)
                    upward_state_messages[k] = np.exp(summand_1 + summand_2)
                if k < self.time_horizon - 1:
                    if k < self.current_action_timestep:
                        #downward_action_messages[k] = self.action_beliefs[k]
                        downward_action_messages[k] = self.U
                    else:
                        downward_action_messages[k] = self.U

            # message backward sweep (T-2 to 0)
            for k in reversed(range(self.time_horizon - 1)):
                if k == self.time_horizon - 2:
                    upward_action_messages[k] = np.einsum('ijk,j,j,i->k', self.B, rightward_state_messages[k], upward_state_messages[k], upward_state_messages[k + 1])
                    leftward_state_messages[k] = np.einsum('ijk,k,i->j', self.B, downward_action_messages[k], upward_state_messages[k + 1])
                else:
                    upward_action_messages[k] = np.einsum('ijk,j,j,i,i->k', self.B, rightward_state_messages[k], upward_state_messages[k], upward_state_messages[k + 1], leftward_state_messages[k + 1])
                    leftward_state_messages[k] = np.einsum('ijk,k,i,i->j', self.B, downward_action_messages[k], upward_state_messages[k + 1], leftward_state_messages[k + 1])

            # belief updating
            for k in range(self.time_horizon):
                if k == self.time_horizon - 1:
                    try:
                        self.state_beliefs[k] = optimize_q_s_iterative(self.alpha, self.C, rightward_state_messages[k])
                    except RuntimeError:
                        self.state_beliefs[k] = (rightward_state_messages[k] * upward_state_messages[k]) / np.sum(rightward_state_messages[k] * upward_state_messages[k])
                else:
                    try:
                        self.state_beliefs[k] = optimize_q_s_iterative(self.alpha, self.C, rightward_state_messages[k], leftward_state_messages[k])
                    except RuntimeError:
                        self.state_beliefs[k] = (rightward_state_messages[k] * upward_state_messages[k] * leftward_state_messages[k]) / np.sum(rightward_state_messages[k] * upward_state_messages[k] * leftward_state_messages[k])   
            for k in range(self.current_observation_timestep + 1, self.time_horizon):
                #self.observation_beliefs[k] = np.einsum('ij,j->i', self.A, self.state_beliefs[k])
                self.observation_beliefs[k] = np.einsum('ij,j->i', self.alpha / np.sum(self.alpha, axis=0), self.state_beliefs[k])
            #for k in range(self.time_horizon - 1):
                #self.action_beliefs[k] = (downward_action_messages[k] * upward_action_messages[k]) / np.sum(downward_action_messages[k] * upward_action_messages[k])

            #current_free_energy = self.free_energy(rightward_state_messages, upward_state_messages, downward_action_messages, upward_action_messages, leftward_state_messages)
        
        #return current_free_energy


    def act(self, action = None):
        if action == None:
            current_action_belief = self.action_beliefs[self.current_action_timestep]
            number_actions = len(current_action_belief)
            action = np.random.choice(number_actions, p=current_action_belief)
        #self.action_beliefs[self.current_action_timestep, :] = 0
        #self.action_beliefs[self.current_action_timestep, action] = 1
        self.current_action_timestep += 1
        return action
    

    def print_beliefs(self):
        for k in range(self.time_horizon):
            print('Timestep ' + str(k))
            if k == self.time_horizon - 1:
                print('q(s) = ' + str(self.state_beliefs[k]))
                print('q(o) = ' + str(self.observation_beliefs[k]))
            else:
                print('q(s) = ' + str(self.state_beliefs[k]))
                print('q(o) = ' + str(self.observation_beliefs[k]))
                print('q(u) = ' + str(self.action_beliefs[k]))
            print()
