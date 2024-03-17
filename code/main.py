import SatteliteMovement as sm
import numpy             as np

import matplotlib.pyplot as plt

if __name__ == "__main__":

    st = sm.TrajectorySolver(400)

    st.setTimeLimits(0., 500 * 60., 5000)
    st.setStartVelocity(np.sqrt(st.mu / st.r))
    st.setRotMatrixAngles(np.pi / 3, 0, 0)
    # st.setRotMatrixAngles(0, 0, np.pi / 3)

    x0 = st.getXVInISK()
    A  = st.getRotMatrix()

    x0[:3] = np.dot(A, x0[:3])
    x0[3:] = np.dot(A, x0[3:])

    t, x1 = st.RK4(st.f_tx, x0)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.plot(x1[0, :], x1[1, :], x1[2, :])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Траектория спутника в ИСК')

    plt.show()

