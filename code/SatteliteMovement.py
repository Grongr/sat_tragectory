from BasePlotting import draw_sphere

import numpy             as np
import numpy.linalg      as lng

import matplotlib.pyplot as plt

mu = 398600.4415 * 1e9
RE = 6378e3
J2 = 1082.6 * 1e-6

class TrajectorySolver:

    x0 = np.zeros(6)
    
    def __init__(self, r, angles, v="1st_cosmic", tlimits=[0, 0, 0]):
        self.r = RE + r * 1e3
        
        if v == "1st_cosmic":
            self.v = np.sqrt(mu / self.r)
        else:
            self.v = v
        
        self.tstart = tlimits[0]
        self.tend   = tlimits[1]
        self.tstep  = tlimits[2]
        
        self.i = angles[0]
        self.o = angles[1]
        self.u = angles[2]
        
        self.calcXVInISK()

    def calcXVInISK(self):
        self.x0 = np.array([0., self.r, 0., -self.v, 0., 0.])
        
        A = self.getRotMatrix()
        self.x0[:3] = np.dot(A, self.x0[:3])
        self.x0[3:] = np.dot(A, self.x0[3:])
        
    def setTimeLimits(self, tlimits):
        self.tstart = tlimits[0]
        self.tend   = tlimits[1]
        self.tstep  = tlimits[2]
        
    def getT(self):
        return (2 * np.pi * self.r / self.v)

    def f_tx(t, x):

        l = lng.norm(x[:3])
        Z = np.array([0, 0, x[2]])

        delta = (3/2) * J2 * mu * (RE**2)

        dxdt = np.zeros(6)
        dxdt[:3] = x[3:]
        dxdt[3:] = -mu * x[:3] / l**3 + delta / (l**5) * ((5 * x[2]**2 / l**2 - 1) * x[:3] - 2 * Z)

        return dxdt

    def RK4(self, f = f_tx):

        t = np.linspace(self.tstart, self.tend, self.tstep)
        h = t[1] - t[0]
        x = np.zeros((self.x0.shape[0], self.tstep))
        x[:, 0] = self.x0

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

        A = np.array([[cos(o)*cos(u) - sin(o)*cos(i)*sin(u), -sin(u)*cos(o) - sin(o)*cos(i)*cos(u), sin(i)*sin(o)],
                      [sin(o)*cos(u) + cos(o)*cos(i)*sin(u), -sin(o)*sin(u) + cos(o)*cos(i)*cos(u), -cos(o)*sin(i)],
                      [sin(i)*sin(u)                       , sin(i)*cos(u)                        , cos(i)]])

        return A

    def draw_track(self, x, t):

        n = np.zeros(self.tstep)
        for i in range(self.tstep):
            n[i] = lng.norm(x[:3, i])

        omega = 2 * np.pi / 24 / 60 / 60
        L = np.arctan2(x[1, :], x[0, :]) - omega * t
        B = np.arcsin(x[2, :] / n)
        
        for i in range(len(L)):

            if L[i] < -np.pi:
                L[i] += 2 * np.pi 

        plt.scatter(L, B, marker='.')
        plt.show()
        
    def plot_sat_traj(self, x, t):
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x[0, :], x[1, :], x[2, :],
                label="Траектория первого КА")
        
        draw_sphere(ax, RE, 5)
        
        lim1 = abs(x).max()
        
        ax.set_xlim(-lim1, lim1)
        ax.set_ylim(-lim1, lim1)
        ax.set_zlim(-lim1, lim1)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        ax.legend()
        
        return ax

