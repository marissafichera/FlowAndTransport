import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

poise_to_microP = 10**7


def fracture_flow():
    b = 0.002   # 2 mm fracture
    mu = 0.001   #viscosity of water in Pascal seconds
    dh = 5         # initial head change
    L = 10          # initial length
    h_grad = dh/L       # initial head gradient


    u_mean = b**2 * h_grad / (3 * mu)
    q = u_mean * b

    y = np.linspace(-b, b, 1000)
    u = (1 / (2*mu))*h_grad*(b**2 - y**2)
    tau_yx = h_grad*y

    plt.figure(1)
    plt.plot(y, u)
    plt.xlabel('y (m)')
    plt.ylabel('velocity (m/s)')
    plt.title('Velocity profile for base case')

    plt.figure(2)
    plt.plot(y, tau_yx)
    plt.xlabel('y (m)')
    plt.ylabel('tau (Pa)')
    plt.title('shear stress profile for base case')
    plt.show()



def main():


    fracture_flow()
    # overland_runoff()

if __name__ == '__main__':
    main()
