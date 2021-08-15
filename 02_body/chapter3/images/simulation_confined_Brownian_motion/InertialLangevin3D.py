# Élodie Millan
# June 2020
# Langevin equation 3D bulk for a free particule with inertia.

import numpy as np
import matplotlib.pyplot as plt

from OverdampedLangevin3D import Langevin3D


class InertialLangevin3D(Langevin3D):
    def __init__(self, dt, Nt, R, rho, eta=0.001, T=300, x0=(0, 0, 0)):
        """

        :param dt: Time step [s].
        :param Nt: Number of time points.
        :param R: Radius of particule [m].
        :param rho: Volumic mass of the particule [kg/m³]
        :param eta: Fluid viscosity (default = 0.001 [Pa/s]).
        :param T: Temperature (default = 300 [k]).
        :param x0: Initial position of particule (default = (0,0,0) [m]).
        """
        super().__init__(dt, Nt, R, eta=eta, T=T, x0=x0)
        self.rho = rho

        self.m = rho * (4 / 3) * np.pi * R ** 3
        self.tau = self.m / self.gamma
        self.a = np.sqrt(2 * self.kb * self.T * self.gamma)  # Coef of the white noise
        self.b = 2 + dt / self.tau
        self.c = 1 + dt / self.tau

    def _PositionXi(self, xi1, xi2, rng):
        """
        Intern methode of InertialLangevin3D class - Position of a Brownian particule at time t.

        :param xi1: Position of the particule at (t - dt).
        :param xi2: Position of the particule at (t - 2dt).
        :param rng: a random number for dBt white noise.
        :return: The position of the particule at time t.
        """
        xi = (
            (self.b / self.c * xi1)
            - (1 / self.c * xi2)
            + (self.a / self.c) * (self.dt**2 / self.m) * rng
        )

        return xi

    def trajectory(self, output=False, Nt=None):

        if Nt == None:
            Nt = self.Nt

        rngx = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=Nt
        )
        rngy = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=Nt
        )
        rngz = (1 / np.sqrt(self.dt)) * np.random.default_rng().normal(
            0.0, 1, size=Nt
        )

        x = np.zeros(Nt)
        y = np.zeros(Nt)
        z = np.zeros(Nt)

        # 2 first values of trajectory compute with random trajectory.
        x[0:2], y[0:2], z[0:2] = super().trajectory(output=True, Nt=2)

        for i in range(2, Nt):

            x[i] = self._PositionXi(x[i - 1], x[i - 2], rngx[i])
            y[i] = self._PositionXi(y[i - 1], y[i - 2], rngy[i])
            z[i] = self._PositionXi(z[i - 1], z[i - 2], rngz[i])

        self.x = x
        self.y = y
        self.z = z

        if output:
            return self.x, self.y, self.z


def test():
    langevin3D = InertialLangevin3D(
        dt=1e-6, Nt=1000000, rho=1050.0, R=1.5e-6, x0=(0.0, 0.0, 0.0)
    )

    langevin3D.trajectory()
    langevin3D.plotTrajectory()
    MSDx = langevin3D.MSD1D("x", output=True)
    MSDy = langevin3D.MSD1D("y", output=True)
    MSDz = langevin3D.MSD1D("z", output=True)

    # ----- MSD 1D -----

    fig1 = plt.figure()
    plt.loglog(
        langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
        MSDx,
        color="red",
        linewidth=0.8,
        label="MSDx inertial",
    )
    plt.loglog(
        langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
        MSDy,
        color="green",
        linewidth=0.8,
        label="MSDy inertial",
    )
    plt.loglog(
        langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
        MSDz,
        color="blue",
        linewidth=0.8,
        label="MSDz inertial",
    )
    plt.plot(
        langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
        (2 * langevin3D.kb * langevin3D.T / langevin3D.gamma)
        * langevin3D.t[langevin3D.list_dt_MSD],
        color="black",
        linewidth=0.8,
        label="Non inertial theory : x = 2D t",
    )
    plt.xlabel("Times t/$ \tau $ [s]")
    plt.ylabel("MSD 1D [m²]")
    plt.title("Mean square displacement 1D")
    plt.legend()
    plt.show()


    # # ----- MSD 3D -----
    #
    # MSD3D = langevin3D.MSD3D(output=True)
    # fig2 = plt.figure()
    # plt.loglog(
    #     langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
    #     MSD3D,
    #     color="red",
    #     linewidth=0.8,
    #     label="Inertial MSD",
    # )
    # plt.plot(
    #     langevin3D.t[langevin3D.list_dt_MSD] / langevin3D.tau,
    #     (6 * langevin3D.kb * langevin3D.T / langevin3D.gamma)
    #     * langevin3D.t[langevin3D.list_dt_MSD],
    #     color="black",
    #     linewidth=0.8,
    #     label="Non inertial theory : x = 6D t",
    # )
    # plt.xlabel("Times $ t/ \tau $")
    # plt.ylabel("MSD 3D [m²]")
    # plt.title("Mean square displacement 1D")
    # plt.legend()
    # plt.show()
    #
    # # langevin3D.speedDistribution1D("x", 10, plot=True)
    # langevin3D.dXDistribution1D("x", 10, plot=True)

if __name__ == '__main__':
    test()