# Discrete Kalman Filter
import sys
sys.path.append('python/')
import numpy as np
from scipy.linalg import expm


class KF:
    ''' Poor Man's Kalman Filter without any checks, controls, etc.
    '''

    def __init__(self, dim, Cin, Qin, Rin):
        self.C = Cin
        self.Q = Qin
        self.R = Rin
        self.dim = dim
        self.Ac = np.zeros((self.dim, self.dim))
        self.x = np.zeros((self.dim,))
        self.P = np.eye((self.dim, self.dim))

    def set_prior(x0, P0):
        self.x = x0
        self.P = P0

    def update(self, y):
        z = y - self.C * self.x
        Theta = self.C * self.P * self.C.transpose + self.R
        K = self.P * self.C.transpose * np.linalg.inv(Theta)
        self.P = self.P - K * self.C * self.P
        self.x = self.x + K * z

    def _discretize(self, dt):
        return expm(self.A * dt)

    def predict(self, dt):
        A = _discretize(dt)
        self.x = A * self.x
        self.P = A * self.P * A + self.Q

    def observe():
        return self.C * self.x
