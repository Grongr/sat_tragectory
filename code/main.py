import SatteliteMovement as sm
import numpy             as np
import numpy.linalg      as lng

import matplotlib.pyplot as plt

if __name__ == "__main__":

    st = sm.TrajectorySolver(400)

    T = 2 * np.pi * st.r / np.sqrt(st.mu / st.r)

    st.setTimeLimits(0., 24 * 60**2, 10000)
    st.setStartVelocity(np.sqrt(st.mu / st.r))
    st.setRotMatrixAngles(np.pi / 180 * 97, 0, np.pi / 6)
    # st.setRotMatrixAngles(0, 0, 0)

    x0 = st.getXVInISK()
    A  = st.getRotMatrix()

    x0[:3] = np.dot(A, x0[:3])
    x0[3:] = np.dot(A, x0[3:])

    t, x1 = st.RK4(st.f_tx, x0)

    x = x1

    n = np.zeros(st.tstep)
    for i in range(st.tstep):
        n[i] = lng.norm(x[:3, i])

    omega = 2 * np.pi / 24 / 60 / 60
    L = np.arctan2(x[1, :], x[0, :]) - omega * t
    B = np.arcsin(x[2, :] / n)

    plt.scatter(L, B, marker='.')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.plot(x[0, :], x[1, :], x[2, :])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    plt.show()

