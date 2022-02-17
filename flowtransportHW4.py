import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import os

def problem_4_3():
    L = 1       # tube length (m)
    Ti = 300        #Initial temp (K)
    T0l = 320        # temp at t=0, left
    T0r = 300       # temp at t = 0, right
    t = np.linspace(0, 100000, 200000)
    x = np.linspace(0, 1, 1000)
    c1 = -20
    c2 = 320

    u = c1*x + c2

    plt.plot(x, u)
    plt.xlabel('x(m)')
    plt.ylabel('temperature (K)')
    plt.title('Initial and final equilibrium temperature profile')
    plt.show()

def problem_4_1():
    P = np.linspace(0.1, 10, 1000)
    atm_to_pa = 101325
    pa_to_kpa = 0.001
    P = P*atm_to_pa
    T = np.linspace(273.15, 100+273.15, 1000)

    R = 287.058   #J/kgK

    rho = P/(R*T)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    X = T
    Y = P
    X, Y = np.meshgrid(X, Y)
    Z = (Y/(R*X))

    surf = ax.plot_surface(X, Y*0.001, Z, cmap=cm.coolwarm)
    plt.xlabel('temperature (kelvin)')
    plt.ylabel('pressure (kPa)')
    fig.colorbar(surf, label='density(kg/m^3)')

    # plt.zlabel('density (kg/m^3)')
    outpath = r'\\agustin\homes\mfichera\My Documents\_phd\classes\hyd508_flowtransport\homework_exercises\HW4'
    # plt.savefig(os.path.join(outpath, 'p41_3dsurf.png'))
    # plt.show()

    rhonot = 1.17       #ref density kg/m^3
    ratios = rho/rhonot
    alphanot = 0.0034      #K^-1
    betanot = 0.00001       # Pa^-1
    Pnot = 100000           # Pa
    Tnot = 298.15           #kelvin

    # print(ratio)
    linapproxrho = rhonot - (rhonot*alphanot*(T - Tnot)) + (rhonot*betanot*(P - Pnot))
    print(linapproxrho)
    larho_ratios = linapproxrho/rhonot

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


if __name__ == '__main__':
    main()