# Élodie Millan
# July 2020
# Langevin equation 3D for a free particule close to a rigid wall with inertia and weight.

import numpy as np
import matplotlib.pyplot as plt

from RigidWallOverdampedLangevin3D import RigidWallOverdampedLangevin3D


class RigidWallInertialLangevin3D(RigidWallOverdampedLangevin3D):  # hérite de RigidWallOverdampedLangevin3D
    def __init__(self, dt, Nt, R, rho, rhoF=1000.0, eta=0.001, T=300.0, x0=None):
        """

        :param dt: float - Time step [s].
        :param Nt: int - Number of time points.
        :param R: float - Radius of particule [m].
        :param rho: float - Volumic mass of the particule [kg/m³].
        :param rhoF: float - Volumic mass of the fluid [kg/m³] (DEFAULT = 1000 [kg/m³]).
        :param eta: float - Fluid viscosity (DEFAULT = 0.001 [Pa/s]).
        :param T: float - Temperature (DEFAULT = 300 [k]).
        :param x0: array float - Initial position of particule (DEFAULT = (0,0,R) [m]).
        """
        if x0 == None:
            x0 = (0.0, 0.0, R)
        super().__init__(dt, Nt, R, rho, eta=eta, T=T, x0=x0)

        self.gamma_mean = 6 * np.pi * eta * R # average of gamma
        self.tau_mean = self.m / self.gamma_mean # average of tau

    def _a(self, gamma):
        """
        Intern methode of RigidWallInertialLangevin3D class - white noise a = sqrt(k T gamma) at t-dt.

        :param gamma: the gamma value used at t-dt times (depends of the coordinate used).

        :return: The white noise a at the position z(t-dt) for a gamma value on x/y or z.
        """

        a = np.sqrt(2 * self.kb * self.T * gamma)

        return a

    def _PositionXi(self, xi_1, xi_2, zi_1, rng, axis=None):
        """
        Intern methode of InertialLangevin3D class - Position of a Brownian particule inertial with rigid wall, at time t.

        :param xi_1: float - Position of the particule at (t - dt).
        :param xi_2: float - Position of the particule at (t - 2dt).
        :param zi_1: float - Perpendicular position by the wall z at (t - dt).
        :param rng: a random number for dBt/dt white noise.
        :param axis: The axis used : put "z" if z axis or None if x/y axis.

        :return: The position of the particule at time t.
        """
        m = self.m
        dt = self.dt
        g = self.g
        delta_m = self.delta_m
        T = self.T
        kb = self.kb
        lD = self.lD

        if axis == "z":
            gamma = self._gamma_z(zi_1)
            weight = -(delta_m / m) * g * dt ** 2
            elec = (4 * kb * T) / (lD * m) * np.exp(-zi_1 / lD) * dt ** 2

        else:
            gamma = self._gamma_xy(zi_1)
            weight = 0
            elec = 0

        # --- Cas dérivée centrée xi+1 - xi-1 /2dt - marche mais différente que le cas précedent
        # b = 1 / (1 + dt * gamma / m)  # coef on factor
        # xi = b * ((dt*gamma/(m) - 1)*xi_2 + 2*xi_1 + 2*weight + 2*elec + 2*self._a(gamma) * dt**2 * rng / m)

        # --- cas dérivée précedent
        b = 2 + dt * gamma / m
        c = 1 + dt * gamma / m
        xi = 1/c * ( b * xi_1 - xi_2 + weight + elec + self._a(gamma) * (dt ** 2 / m) * rng)

        if axis == "z":
            if xi < 0:
                xi = -xi  # reflection

        return xi

    def trajectory(self, output=False):
        """

        :param output: Boolean, if true function output x, y, z (default : false).

        :return: return the x, y, z trajectory.
        """
        rngx = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=self.Nt
        )
        rngy = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=self.Nt
        )
        rngz = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=self.Nt
        )

        x = np.zeros(self.Nt)
        y = np.zeros(self.Nt)
        z = np.zeros(self.Nt)

        # 2 first values of trajectory
        x[0:2] = np.array([self.x0[0], self.x0[0]])
        y[0:2] = np.array([self.x0[1], self.x0[1]])
        z[0:2] = np.array([self.x0[2], self.x0[2]])

        for i in range(2, self.Nt):
            x[i] = self._PositionXi(x[i - 1], x[i - 2], z[i - 1], rngx[i])
            y[i] = self._PositionXi(y[i - 1], y[i - 2], z[i - 1], rngy[i])
            z[i] = self._PositionXi(z[i - 1], z[i - 2], z[i - 1], rngz[i], "z")

        self.x = x
        self.y = y
        self.z = z

        if output:
            return self.x, self.y, self.z

def test():
    langevin3D = RigidWallInertialLangevin3D(
        dt=1e-6, Nt=1000000, R=1.5e-6, rho=1050, x0=(0.0, 0.0, 1.0e-6)
    )
    langevin3D.trajectory()

    # langevin3D.plotTrajectory()
    # MSDx = langevin3D.MSD1D("x", output=True)
    # MSDy = langevin3D.MSD1D("y", output=True)
    # MSDz = langevin3D.MSD1D("z", output=True)
    #
    # # ----- MSD 1D -----
    #
    # fig1 = plt.figure()
    # plt.loglog(
    #     langevin3D.t[langevin3D.list_dt_MSD],
    #     MSDx,
    #     color="red",
    #     linewidth=0.8,
    #     label="MSDx inertial",
    # )
    # plt.loglog(
    #     langevin3D.t[langevin3D.list_dt_MSD],
    #     MSDy,
    #     color="green",
    #     linewidth=0.8,
    #     label="MSDy inertial",
    # )
    # plt.loglog(
    #     langevin3D.t[langevin3D.list_dt_MSD],
    #     MSDz,
    #     color="blue",
    #     linewidth=0.8,
    #     label="MSDz inertial",
    # )
    # plt.plot(
    #     langevin3D.t[langevin3D.list_dt_MSD],
    #     (2 * langevin3D.kb * langevin3D.T / langevin3D.gamma_mean)
    #     * langevin3D.t[langevin3D.list_dt_MSD],
    #     color="black",
    #     linewidth=0.8,
    #     label="Non inertial theory : x = 2D t",
    # )
    # plt.xlabel("Times t [s]")
    # plt.ylabel("MSD 1D [m²]")
    # plt.title("Mean square displacement 1D")
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    test()