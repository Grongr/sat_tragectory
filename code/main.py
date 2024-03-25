import SatteliteMovement as sm
import SatteliteGroup    as sg
import numpy             as np
import numpy.linalg      as lng

import matplotlib.pyplot as plt

def initTrajSolver():

    angles = [np.pi / 180 * 97, 0, np.pi / 6]
    st = sm.TrajectorySolver(400, angles, v="1st_cosmic")
                             
    T = st.getT()
    
    tlimits = [0, 1 * T, 10000]
    st.setTimeLimits(tlimits)
    
    return st

if __name__ == "__main__":

    st = initTrajSolver()
    
    group = sg.GroupTrajectory(40)
    
    group.nStartingVectors(st)
    
    plt.show()
    
