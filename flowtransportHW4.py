import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import os


def problem_4_3():
    L = 1  # tube length (m)
    # Ti = 300        #Initial temp (K)
    T0l = 320  # temp at t=0, left
    T0r = 300  # temp at t = 0, right
    # t = np.linspace(0, 100000, 200000)
    x = np.linspace(0, 1, 1000)
    c1 = -20
    c2 = 320

    tinitial = T0r + (T0r - T0r) * x  # uniform initial temp distribution

    T_f = c1 * x + c2
    plt.figure(1)
    plt.plot(x, tinitial, label='initial')
    plt.plot(x, T_f, label='final')
    plt.xlabel('x(m)')
    plt.ylabel('temperature (K)')
    plt.title('Initial and final equilibrium temperature profiles')
    plt.legend()
    plt.show()

    tfinal = 100
    dt = 0.001
    timesteps = np.linspace(0, tfinal, 101)
    print(timesteps)
    num_nvals = 10
    num_tvals = 10
    # entire matrix of n values for each timestep
    # m = np.zeros((num_tvals, num_nvals))
    m = []
    eterm = np.zeros((num_nvals))
    print(m)
    ss = []
    # print(np.arange(1, 100000))
    temp_profiles = np.zeros((num_tvals, len(x))

    # summation of sin terms

    K = 0.000000155

    for t in timesteps:
        print('timestep = ', t)
        for n in np.arange(1, num_nvals+1):
            print('n value = ', n)
            eterm = -K * (n ** 2) * (np.pi ** 2) * t / (L ** 2)
            print('eterm = ', eterm)
            sumterm = (1 / n) * np.sin(n * np.pi * x / L) * np.exp(eterm)
            print('sumterm = ', sumterm)
            # print(sumterm)
        # print('SUM = ', np.sum(sumterm))
        temp = T0l + (T0r - T0l)*(x/L) + (T0r - T0l)*(2/np.pi)*np.sum(sumterm)

        # print('temp at t={} = '.format(t), temp)

        # m[t] = np.sum(sumterm)
        # print('m = ', m)

    #     # print('m = ', m)
    #     s = np.sum(m)
    #     ss.append(s)
    # print(ss)
    # T_x = T0l + (T0r - T0l)*x + (2/np.pi)*ss*(T0r-T0l)
    #
    # plt.figure(2)
    # plt.plot(T_x)

    # plt.figure(2)
    # plt.plot(m)

    # plt.figure(2)
    # plt.plot(x, ms)


def problem_4_1():
    P = np.linspace(0.1, 10, 1000)
    atm_to_pa = 101325
    pa_to_kpa = 0.001
    P = P * atm_to_pa
    T = np.linspace(273.15, 100 + 273.15, 1000)

    R = 287.058  # J/kgK

    rho = P / (R * T)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    X = T
    Y = P
    X, Y = np.meshgrid(X, Y)
    Z = (Y / (R * X))

    surf = ax.plot_surface(X, Y * 0.001, Z, cmap=cm.coolwarm)
    plt.xlabel('temperature (kelvin)')
    plt.ylabel('pressure (kPa)')
    fig.colorbar(surf, label='density(kg/m^3)')

    # plt.zlabel('density (kg/m^3)')
    outpath = r'\\agustin\homes\mfichera\My Documents\_phd\classes\hyd508_flowtransport\homework_exercises\HW4'
    # plt.savefig(os.path.join(outpath, 'p41_3dsurf.png'))
    # plt.show()

    rhonot = 1.17  # ref density kg/m^3
    ratios = rho / rhonot
    alphanot = 0.0034  # K^-1
    betanot = 0.00001  # Pa^-1
    Pnot = 100000  # Pa
    Tnot = 298.15  # kelvin

    # print(ratio)
    linapproxrho = rhonot - (rhonot * alphanot * (T - Tnot)) + (rhonot * betanot * (P - Pnot))
    print(linapproxrho)
    larho_ratios = linapproxrho / rhonot

    tr = []
    pr = []
    for ratio, temp, pressure in zip(larho_ratios, T, P):
        if (ratio <= 1.02) & (ratio >= 0.98):
            tr.append(temp)
            pr.append(pressure)
        else:
            print('out of range')

    mtr = np.min(tr)
    mpr = np.min(pr)
    mxtr = np.max(tr)
    mxpr = np.max(pr)

    print('min temp, pressure = ', mtr, 'and', mpr)
    print('max temp, pressure = ', mxtr, 'and', mxpr)


def main():
    problem_4_3()

    outpath = r'\\agustin\homes\mfichera\My Documents\_phd\classes\hyd508_flowtransport\homework_exercises\HW4'
    # for i in np.arange(however many figs)
    #     plt.savefig(os.path.join(outpath, 'p4_fig().png'.format(i)))


if __name__ == '__main__':
    main()
