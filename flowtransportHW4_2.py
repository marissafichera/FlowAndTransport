import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import os


def problem_4_3():
    L = 0.5  # tube length (m)
    T0l = 320  # temp at t=0, left
    T0r = 300  # temp at t = 0, right
    x = np.linspace(0, 0.5, 100)
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
    # plt.show()

    tfinal = 5
    dt = 0.001
    timesteps = np.linspace(0, tfinal, 50)
    timesteps = [0, 50000, 100000, 500000, 1000000, 10000000000000]
    # print(timesteps)
    num_nvals = 100
    num_tvals = 100
    # entire matrix of n values for each timestep
    # m = np.zeros((num_tvals, num_nvals))
    m = []
    eterm = np.zeros((num_nvals))
    # print(m)
    ss = []
    # print(np.arange(1, 100000))
    lx = len(x)
    lt = len(timesteps)
    temp_profile = np.zeros((lt, lx))
    temp_at_x_at_t = np.zeros(lx)

    # summation of sin terms
    Kwater = 0.629
    D = 0.000000155
    for t in timesteps:
        print('timestep = ', t)
        temps_x = []
        qs_x = []
        for h in x:
            sumterms = []
            for n in np.arange(1, num_nvals+1):
                eterm = -D * (n ** 2) * (np.pi ** 2) * t / (L ** 2)
                sumterm = (1 / n) * np.sin(n * np.pi * h / L) * np.exp(eterm)
                sumterms.append(sumterm)
            sum = np.sum(sumterms)
            print('h = ', h)
            print(sum)
            temp_at_x_at_t = T0l + (T0r - T0l) * (h / L) + (T0r - T0l) * (2 / np.pi) * sum
            q = -Kwater * temp_at_x_at_t
            temps_x.append(temp_at_x_at_t)
            qs_x.append(q)
            print('temp at x at t = ', temp_at_x_at_t)
            print('q = ', q)
        plt.figure(4)
        plt.plot(x, temps_x, label='t={} s'.format(t))
        plt.legend()
        plt.xlabel('x(m)')
        plt.ylabel('temperature (K)')
        plt.title('Temperature profile of heat transfer with time, x = 0.5m')
        plt.figure(6)
        plt.scatter(t, qs_x[0], label='t = {}'.format(t))
        plt.legend()
        plt.xlabel('time (s)')
        plt.ylabel('q ()')
        plt.title('heat flux transient')


    plt.show()

    Kwater = 0.629      #in W/mK
    heatfluxdensityfinal = -Kwater*T_f
    heatfluxdensityinitial = -Kwater*tinitial
    plt.figure(5)
    plt.plot(x, heatfluxdensityinitial, label='initial')
    plt.plot(x, heatfluxdensityfinal, label='final')
    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('heat flux density (W/m^2)')
    plt.title('Heat flux density at initial and final steady states')
    plt.show()

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
