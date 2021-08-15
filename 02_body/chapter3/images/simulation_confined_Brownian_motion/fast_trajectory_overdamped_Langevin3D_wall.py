import numpy as np
from numba import jit

@jit
def Trajectory(Nt, R, delta_m, eta, dt, lD, x0):

    rngx = (1 / np.sqrt(dt)) * np.random.default_rng().normal(0.0, 1, size=Nt)
    rngy = (1 / np.sqrt(dt)) * np.random.default_rng().normal(0.0, 1, size=Nt)
    rngz = (1 / np.sqrt(dt)) * np.random.default_rng().normal(0.0, 1, size=Nt)


    x = np.zeros(Nt)
    y = np.zeros(Nt)
    z = np.zeros(Nt)

    x[0] = x0[0]
    y[0] = x0[1]
    z[0] = x0[2]

    for i in range(1, Nt):
        x[i] = positionxi(x[i - 1], z[i - 1], R, rngx[i], delta_m, dt, eta, lD)
        y[i] = positionxi(y[i - 1], z[i - 1], R, rngy[i], delta_m, dt, eta, lD)
        z[i] = positionxi(z[i - 1], z[i - 1], R, rngz[i], delta_m, dt, eta, lD, "z")

    return x, y, z

@jit
def positionxi(xi_1, zi_1, R, rng, delta_m, dt, lD, eta, axis=None):
    g = 9.81
    kb = 1.380e-23
    T = 300

    if axis == "z":
        gamma = _gamma_z(zi_1, R, eta)
        weight = delta_m * g * dt / (gamma)
        elec = 4*kb*T/lD * np.exp(-zi_1 / lD) * dt / gamma
        correction = (
            kb
            * T
            * (42 * R * zi_1 ** 2 + 24 * R ** 2 * zi_1 + 4 * R ** 3)
            / (
                (6 * zi_1 ** 2 + 9 * R * zi_1 + 2 * R ** 2)
                * (6 * zi_1 ** 2 + 2 * R * zi_1)
            )
            * dt
            / gamma
        )

    else:
        gamma = _gamma_xy(zi_1, R, eta)
        elec = 0
        weight = 0
        correction = 0

    xi = xi_1 - weight + elec + correction + _a(gamma) * rng * dt

    if axis == "z":
        if xi <= 0:
            xi = -xi

    return xi
@jit
def _gamma_z(zi_1, R, eta):
    """
    Intern methode of RigidWallInertialLangevin3D class - gamma on z at time t-dt.

    :param zi_1: float - Perpendicular position by the wall z at (t - dt).

    :return: float - gamma_z = 6πη(z)R : the gamma value for z trajectory dependant of z(t-dt).
    """
    # Padé formula
    gamma_z = (
        6
        * np.pi
        * R
        * eta
        * (
            (
                (6 * zi_1 ** 2 + 2 * R * zi_1)
                / (6 * zi_1 ** 2 + 9 * R * zi_1 + 2 * R ** 2)
            )
            ** (-1)
        )
    )

    return gamma_z
@jit
def _gamma_xy(zi_1, R, eta):
    """
    Intern methode of RigidWallInertialLangevin3D class - gamma on x and y at time t-dt.

    :param zi_1: float - Perpendicular position by the wall z at (t - dt).

    :return: gamma_x = gamma_y = 6πη(z)R : the gamma value for x and y trajectories dependant of z(t-dt).
    """
    # Libchaber formula
    gamma_xy = (
        6
        * np.pi
        * R
        * eta
        * (
            1
            - ((9 * R) / (16 * (zi_1 + R)))
            + (R / (8 * (zi_1 + R))) ** 3
            - (45 * R / (256 * (zi_1 + R))) ** 4
            - (R / (16 * (zi_1 + R))) ** 5
        )
        ** (-1)
    )
    return gamma_xy
@jit
def _a(gamma):
    """
    Intern methode of RigidWallInertialLangevin3D class - white noise a = sqrt(k T gamma) at t-dt.

    :param zi_1: float - Perpendicular position by the wall z at (t - dt).
    :param gamma: the gamma value used (depends of the coordinate used).

    :return: The white noise a at the position z(t-dt) for a gamma value on x/y or z.
    """
    kb = 1.380e-23
    T = 300
    a = np.sqrt(2 * kb * T / gamma)
    return a


if __name__ == "__main__":
    import time
    Nt = 10000000
    delta_m = 7.068583470577035e-16
    R = 1.5e-6
    eta = 0.001
    dt = 1 / 60
    lD = 70e-9
    x0 = (0, 0, 1.5e-6)
    t = time.time()
    x, y, z = Trajectory(Nt, R, delta_m, eta, dt, lD, x0)
    print(time.time() - t)


    Nt = 10000000
    t = time.time()
    x, y, z = Trajectory(Nt, R, delta_m, eta, dt, lD, x0)
    print(time.time() - t)
