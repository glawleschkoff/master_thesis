import numpy as np
from utils import softmax


class GenerativeModel:
    def __init__(self, a, c):
        # self.NUM_OBSERVATIONS = 16
        self.A = np.zeros((16, 8))
        self.A[:2, :2] = np.ones((2, 2)) * 0.5
        self.A[6:8, 2:4] = np.array([[a, 1 - a], [1 - a, a]])
        self.A[10:12, 4:6] = np.array([[1 - a, a], [a, 1 - a]])
        self.A[12, 6] = 1.0
        self.A[13, 7] = 1.0

        self.C = np.zeros((16, 1))
        self.C[2] = c
        self.C[3] = -c
        self.C[6] = c
        self.C[7] = -c
        self.C[10] = c
        self.C[11] = -c
        self.C[14] = c
        self.C[15] = -c
        self.C = softmax(self.C)

        self.D = np.zeros((8, 1))
        self.D[0] = 0.5
        self.D[1] = 0.5
        self.D = softmax(self.D)

    def B(self, u):
        B = np.zeros((8, 8))
        B[2:4, 2:4] = np.eye(2)
        B[4:6, 4:6] = np.eye(2)

        if u == 0:
            B[0, 0] = 1.0
            B[1, 1] = 1.0
            B[0, 6] = 1.0
            B[1, 7] = 1.0
        elif u == 1:
            B[2, 0] = 1.0
            B[3, 1] = 1.0
            B[2, 6] = 1.0
            B[3, 7] = 1.0
        elif u == 2:
            B[4, 0] = 1.0
            B[5, 1] = 1.0
            B[4, 6] = 1.0
            B[5, 7] = 1.0
        elif u == 3:
            B[6, 0] = 1.0
            B[7, 1] = 1.0
            B[6, 6] = 1.0
            B[7, 7] = 1.0
        else:
            raise ValueError(f"Invalid action 'u': {u}. Expected 0, 1, 2, or 3.")

        return B
