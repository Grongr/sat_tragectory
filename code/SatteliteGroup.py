import SatteliteMovement as sm
import numpy             as np
import numpy.linalg      as lng

import matplotlib.pyplot as plt

class GroupTrajectory:

    def __init__(self, sat_count):
        self.sat_count = sat_count
        
    def nStartingVectors(self, st):
        
        t1, x1 = st.RK4()
        x = x1.copy()
        t = t1.copy()
        
        Ar = lng.inv(st.getRotMatrix())
        x[:3] = np.dot(Ar, x[:3])
        x[3:] = np.dot(Ar, x[3:])
        
        T = st.getT()
        t = t[t <= T]
        ts = T / self.sat_count
        ht = t[1] - t[0]
        
        dang = 2 * np.pi / self.sat_count
        hang = 2 * np.pi / T * ht
        
        # Rotating vector in OXY
        phi = np.arctan2(x[1, :t.shape[0]],
                         x[0, :t.shape[0]])
        
        phi += np.pi
                   
        phies = []
        
        for i in range(self.sat_count):
            j = np.where( abs(abs(phi - i*dang) - hang) ==
                          abs(abs(phi - i*dang) - hang).min() )
            phies.append(j[0][0])
        
        print(len(phies))
        print(phies)
        print(phi[2500])
        starting_points = np.zeros( (6, self.sat_count) )
        
        for i in range(self.sat_count):
            starting_points[:, i] = x1[:, phies[i]]
            
        ax1 = st.plot_sat_traj(x1, t1)
        ax1.scatter(starting_points[0, :],
                    starting_points[1, :],
                    starting_points[2, :],
                    label="Стартовые точки n спутников")
        
