import numpy as np

# hidden state names
s = {
    'O_RL' : 0, 'O_RR' : 1, 'C_RL' : 2, 'C_RR' : 3, 'L_RL' : 4, 'L_RR' : 5, 'R_RL' : 6, 'R_RR' : 7
}

# observation names
o = {
    'O_CL' : 0, 'O_CR' : 1, 'O_RW' : 2, 'O_NR' : 3, 'C_CL' : 4, 'C_CR' : 5, 'C_RW' : 6, 'C_NR' : 7,
    'L_CL' : 8, 'L_CR' : 9, 'L_RW' : 10, 'L_NR' : 11, 'R_CL' : 12, 'R_CR' : 13, 'R_RW' : 14, 'R_NR' : 15
}

# action names
u = {
    'O' : 0, 'C' : 1, 'L' : 2, 'R' : 3
}

def generate_gfe_params(alpha=0.9, c=2):
    # A-Matrix "p(o|s)" (likelihood mapping from hidden states to observations)
    A = np.zeros((16, 8))
    A[o['O_CL'], s['O_RL']] = 0.5
    A[o['O_CL'], s['O_RR']] = 0.5
    A[o['L_RW'], s['L_RL']] = alpha
    A[o['L_RW'], s['L_RR']] = 1 - alpha
    A[o['R_RW'], s['R_RL']] = 1 - alpha
    A[o['R_RW'], s['R_RR']] = alpha
    A[o['C_CL'], s['C_RL']] = 1
    A[o['C_CR'], s['C_RR']] = 1
    A[o['O_CR'], s['O_RL']] = 0.5
    A[o['O_CR'], s['O_RR']] = 0.5
    A[o['L_NR'], s['L_RL']] = 1 - alpha
    A[o['L_NR'], s['L_RR']] = alpha
    A[o['R_NR'], s['R_RL']] = alpha
    A[o['R_NR'], s['R_RR']] = 1 - alpha

    # B-Tensor "p(s_next|s_current, u_next)" (hidden states' transition probability given an action)
    B = np.zeros((8, 8, 4))
    B[s['O_RL'], s['O_RL'], u['O']] = 1
    B[s['O_RL'], s['C_RL'], u['O']] = 1
    B[s['O_RL'], s['R_RL'], u['O']] = 1
    B[s['O_RL'], s['L_RL'], u['O']] = 1
    B[s['C_RL'], s['O_RL'], u['C']] = 1
    B[s['C_RL'], s['C_RL'], u['C']] = 1
    B[s['O_RL'], s['R_RL'], u['C']] = 1
    B[s['O_RL'], s['L_RL'], u['C']] = 1
    B[s['L_RL'], s['O_RL'], u['L']] = 1
    B[s['L_RL'], s['C_RL'], u['L']] = 1
    B[s['O_RL'], s['R_RL'], u['L']] = 1
    B[s['O_RL'], s['L_RL'], u['L']] = 1
    B[s['R_RL'], s['O_RL'], u['R']] = 1
    B[s['R_RL'], s['C_RL'], u['R']] = 1
    B[s['O_RL'], s['R_RL'], u['R']] = 1
    B[s['O_RL'], s['L_RL'], u['R']] = 1
    B[s['O_RR'], s['O_RR'], u['O']] = 1
    B[s['O_RR'], s['C_RR'], u['O']] = 1
    B[s['O_RR'], s['R_RR'], u['O']] = 1
    B[s['O_RR'], s['L_RR'], u['O']] = 1
    B[s['C_RR'], s['O_RR'], u['C']] = 1
    B[s['C_RR'], s['C_RR'], u['C']] = 1
    B[s['O_RR'], s['R_RR'], u['C']] = 1
    B[s['O_RR'], s['L_RR'], u['C']] = 1
    B[s['L_RR'], s['O_RR'], u['L']] = 1
    B[s['L_RR'], s['C_RR'], u['L']] = 1
    B[s['O_RR'], s['R_RR'], u['L']] = 1
    B[s['O_RR'], s['L_RR'], u['L']] = 1
    B[s['R_RR'], s['O_RR'], u['R']] = 1
    B[s['R_RR'], s['C_RR'], u['R']] = 1
    B[s['O_RR'], s['R_RR'], u['R']] = 1
    B[s['O_RR'], s['L_RR'], u['R']] = 1

    # C-Vector "p~(o)" (goal distribution at each timestep)
    C = np.ones(16)
    C = C * (1 / (4 * np.exp(c) + 4 * np.exp(-c) + 8))
    C[o['O_RW']] = np.exp(c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['C_RW']] = np.exp(c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['L_RW']] = np.exp(c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['R_RW']] = np.exp(c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['O_NR']] = np.exp(-c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['C_NR']] = np.exp(-c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['L_NR']] = np.exp(-c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)
    C[o['R_NR']] = np.exp(-c) / (4 * np.exp(c) + 4 * np.exp(-c) + 8)

    # D-Vector "p(s_0)" (initial hidden state distribution)
    D = np.zeros(8)
    D[s['O_RL']] = 0.5
    D[s['O_RR']] = 0.5

    # U-Vector "p(u)" (prior distribution for actions)
    U = np.ones(4) * 0.25
    
    return A, B, C, D, U