import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

poise_to_microP = 10 ** 7
outpath = r'/Users/marissafichera/Documents/classes/hyd508/HW3'


# def plot_tau_profile(n, type, x, y, slope, forcing):
#
#     y = np.linspace(0, h, 1000)
#     theta = np.linspace(theta_base, theta_max, 10)
#
#     plt.figure(n + 100)
#     for i in np.arange(y):
#         tau_yx = rho * g * np.sin(theta) * y
#         plt.plot(y, tau_yx)




def plot_mean_velocity(n, type, x, y, forcing):
    plt.figure(n)
    plt.plot(x, y)
    plt.xlabel('{}'.format(forcing))
    plt.ylabel('mean velocity (m/s)')
    plt.title('{} - mean velocity as function of {}'.format(type, forcing))
    plt.savefig(os.path.join(outpath, 'plot_{}.png'.format(n)))


def change_slope(h, mu, rho, g, theta_base, b, h_grad_base, type):
    if type == 'fracture':
        h_grad_max = 10 * h_grad_base
        h_grad = np.linspace(h_grad_base, h_grad_max, 1000)

        u_mean = b ** 2 * h_grad / (3 * mu)
        q = u_mean * b

        y = np.linspace(-b, b, 1000)

        # shear profile - fracture
        tau_yx = h_grad * y

        # velocity profile - fracture
        u = (1 / (2 * mu)) * h_grad * (b ** 2 - y ** 2)

        slope = h_grad
        n = 5
        sl = np.linspace(np.min(slope), np.max(slope), 10)

        plt.figure(n + 100)
        for s in sl:
            tau_yx = s * y
            plt.plot(y, tau_yx, label='{}'.format(s))
            plt.xlabel('aperture (m)')
            plt.ylabel('shear stress (Pa)')
            plt.title('{}: shear stress profile as function of gradient'.format(type))
        plt.legend()
        plt.savefig(os.path.join(outpath, 'plot_shear_gradient_fracture.png'))


    elif type == 'runoff':
        deg_max = 80  # max slope in degrees... basically a waterfall and physically impossible
        theta_max = deg_max * np.pi / 180
        theta = np.linspace(theta_base, theta_max, 1000)

        u_mean = h ** 2 * rho * g * np.sin(theta) / (3 * mu)
        q = u_mean * h

        y = np.linspace(0, h, 1000)

        # shear profile
        tau_yx = rho * g * np.sin(theta) * y

        # velocity profile
        u = rho * g * np.sin(theta) * (h ** 2 - y ** 2) / (2 * mu)

        slope = theta
        n = 6
        sl = np.linspace(np.min(slope), np.max(slope), 10)

        plt.figure(n + 100)
        for s in sl:
            tau_yx = rho * g * np.sin(s) * y
            plt.plot(y, tau_yx, label='{}'.format(s))
            plt.xlabel('depth (m)')
            plt.ylabel('shear stress (Pa)')
            plt.title('{}: shear stress profile as function of slope'.format(type))
        plt.legend()
        plt.savefig(os.path.join(outpath, 'plot_shear_slope_runoff.png'))

    plot_mean_velocity(n, type, slope, u_mean, forcing='gradient or slope')
    # plot_tau_profile(n, type, y, tau_yx, slope, forcing='gradient or slope')
    # y = np.linspace(0, h, 1000)

    # slope = np.linspace(np.min(slope), np.max(slope), 10)
    #
    # plt.figure(n + 100)
    # for s in slope:
    #     tau_yx = rho * g * np.sin(s) * y
    #     plt.plot(y, tau_yx, label='{}'.format(s))
    #     plt.xlabel('Aperture or depth (m)')
    #     plt.ylabel('shear stress (Pa)')
    #     plt.title('{}: shear stress profile as function of gradient/slope'.format(type))


def change_h(h_base, mu, rho, g, theta, b_base, h_grad, type):
    if type == 'fracture':
        b_max = 0.01  # 10 centimeter wide aperture
        b = np.linspace(b_base, b_max, 1000)

        u_mean = b ** 2 * h_grad / (3 * mu)
        q = u_mean * b

        y = np.linspace(-b, b, 1000)

        # shear profile - fracture
        tau_yx = h_grad * y

        # velocity profile - fracture
        u = (1 / (2 * mu)) * h_grad * (b ** 2 - y ** 2)

        depth = b
        n = 3

    elif type == 'runoff':
        h_max = 2  # 2 meter height runoff (large flood)
        h = np.linspace(h_base, h_max, 1000)

        u_mean = h ** 2 * rho * g * np.sin(theta) / (3 * mu)
        q = u_mean * h

        y = np.linspace(0, h, 1000)

        # shear profile
        tau_yx = rho * g * np.sin(theta) * y

        # velocity profile
        u = rho * g * np.sin(theta) * (h ** 2 - y ** 2) / (2 * mu)

        depth = h
        n = 4

    plot_mean_velocity(n, type, depth, u_mean, forcing='aperture or depth (m)')


def change_viscosity(h, mu_base, rho, g, theta, b, h_grad, type):
    if type == 'fracture':
        mu_max = 10 * mu_base
        mu = np.linspace(mu_base, mu_max, 1000)

        u_mean = b ** 2 * h_grad / (3 * mu)
        q = u_mean * b

        y = np.linspace(-b, b, 1000)

        # shear profile - fracture
        tau_yx = h_grad * y

        # velocity profile - fracture
        u = (1 / (2 * mu)) * h_grad * (b ** 2 - y ** 2)
        n = 1

    elif type == 'runoff':
        mu_max = 10 * mu_base
        mu = np.linspace(mu_base, mu_max, 1000)

        u_mean = h ** 2 * rho * g * np.sin(theta) / (3 * mu)
        q = u_mean * h

        y = np.linspace(0, h, 1000)

        # shear profile
        tau_yx = rho * g * np.sin(theta) * y

        # velocity profile
        u = rho * g * np.sin(theta) * (h ** 2 - y ** 2) / (2 * mu)
        n = 2

    plot_mean_velocity(n, type, mu, u_mean, forcing='viscosity (Pa*s)')
    # plot_tau_profile(type, forcing='viscosity (Pa*s)', y, tau_yx, mu)


def runoff():
    h = 0.002  # 2 mm height of runoff
    mu = 0.001  # viscosity of water Pa*s
    rho = 997  # density of water in kg/m^3
    g = 9.81  # gravity in m/s^2
    deg = 10  # slope in degrees
    theta = deg * np.pi / 180  # slope in radians

    u_mean = h ** 2 * rho * g * np.sin(theta) / (3 * mu)
    q = u_mean * h

    y = np.linspace(0, h, 1000)

    # shear profile
    tau_yx = rho * g * np.sin(theta) * y

    # velocity profile
    u = rho * g * np.sin(theta) * (h ** 2 - y ** 2) / (2 * mu)

    change_viscosity(h, mu, rho, g, theta, '', '', 'runoff')
    change_h(h, mu, rho, g, theta, '', '', 'runoff')
    change_slope(h, mu, rho, g, theta, '', '', 'runoff')


def fracture_flow():
    b = 0.002  # 2 mm fracture
    mu = 0.001  # viscosity of water in Pascal seconds
    dh = 5  # initial head change
    L = 10  # initial length
    h_grad = dh / L  # initial head gradient

    u_mean = b ** 2 * h_grad / (3 * mu)
    q = u_mean * b

    y = np.linspace(-b, b, 1000)
    u = (1 / (2 * mu)) * h_grad * (b ** 2 - y ** 2)
    tau_yx = h_grad * y

    change_viscosity('', mu, '', '', '', b, h_grad, 'fracture')
    change_h('', mu, '', '', '', b, h_grad, 'fracture')
    change_slope('', mu, '', '', '', b, h_grad, 'fracture')

    # plt.figure(1)
    # plt.plot(y, u)
    # plt.xlabel('y (m)')
    # plt.ylabel('velocity (m/s)')
    # plt.title('Velocity profile for base case')
    #
    # plt.figure(2)
    # plt.plot(y, tau_yx)
    # plt.xlabel('y (m)')
    # plt.ylabel('tau (Pa)')
    # plt.title('shear stress profile for base case')
    # plt.show()


def main():
    fracture_flow()
    runoff()
    plt.show()


if __name__ == '__main__':
    main()
