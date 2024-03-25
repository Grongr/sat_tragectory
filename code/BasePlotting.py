import numpy             as np
import numpy.linalg      as lng

import matplotlib.pyplot as plt

def z_rotating(phi):

    M = np.array([[np.cos(phi), -np.sin(phi), 0.],
                  [np.sin(phi),  np.cos(phi), 0.],
                  [         0.,           0., 1.]])
    return M
    
def draw_sphere(ax, r, n):

    phi = np.linspace(0, 2 * np.pi, 100)
    psi = np.linspace(0,     np.pi, 100)

    x_s = r * np.cos(phi)
    y_s = r * np.sin(phi)

    X = np.array([np.zeros(100), x_s, y_s])

    u = np.linspace(0, np.pi, n)

    for i in range(n):
        X1 = np.dot(z_rotating(u[i]), X)
                              
        ax.plot(X1[0, :], X1[1, :], X1[2, :], color="green")
        
