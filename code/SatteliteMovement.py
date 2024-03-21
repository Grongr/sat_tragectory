import matplotlib.pyplot as plt

import numpy as np

import numpy.linalg as lng

class TrajectorySolver:

    mu = 398600.4415 * 1e9
    RE = 6378e3
    J2 = 1082.6 * 1e-6

    def __init__(self, r):
        self.r = self.RE + r * 1e3

    def setTimeLimits(self, tstart, tend, tstep):
        self.tstart = tstart
        self.tend   = tend
        self.tstep  = tstep

    def setStartVelocity(self, v):
        self.v = v

    def setRotMatrixAngles(self, i, o, u):
        self.i = i
        self.o = o
        self.u = u

    def getXVInISK(self):
        return np.array([0., self.r, 0., -self.v, 0., 0.])

    def f_tx(self, t, x):

        l = lng.norm(x[:3])
        Z = np.array([0, 0, x[2]])

        delta = (3/2) * self.J2 * self.mu * (self.RE**2)

        dxdt = np.zeros(6)
        dxdt[:3] = x[3:]
        dxdt[3:] = -self.mu * x[:3] / l**3 + delta / (l**5) * ((5 * x[2]**2 / l**2 - 1) * x[:3] - 2 * Z)

        return dxdt

    def RK4(self, f, x0):

        t = np.linspace(self.tstart, self.tend, self.tstep)
        h = t[1] - t[0]
        x = np.zeros((x0.shape[0], self.tstep))
        x[:, 0] = x0

        for i in range(1, self.tstep):

            k1 = f(t[i-1], x[:, i-1])
            k2 = f(t[i-1] + h * 0.5, x[:, i-1] + h * 0.5 * k1)
            k3 = f(t[i-1] + h * 0.5, x[:, i-1] + h * 0.5 * k2)
            k4 = f(t[i-1] + h, x[:, i-1] + h * k3)

            x[:, i] = x[:, i-1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return t, x

    def getRotMatrix(self):

        o = self.o
        u = self.u
        i = self.i
        cos = np.cos
        sin = np.sin

        A = np.array([[cos(o)*cos(u) - sin(o)*cos(i)*sin(u), -cos(u)*cos(o) - sin(o)*cos(i)*cos(u), sin(i)*sin(o)],
                      [sin(o)*cos(u) + cos(o)*cos(i)*sin(u), -sin(o)*sin(u) + cos(o)*cos(i)*cos(u), -cos(o)*sin(i)],
                      [sin(i)*sin(u)                       , sin(i)*cos(u)                        , cos(i)]])

        return A

