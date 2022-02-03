import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def calc_viscosity(M, k, R, T, p, mc):
    nm_to_m = 10**-18
    g_to_kg = 0.001

    molecular_area = 0.4*nm_to_m
    print('m*mc in kg/mol * mol/= ', M*g_to_kg*mc)

    c = np.sqrt((8*R*T)/(np.pi*M*g_to_kg))
    print('c = ', c)
    l = (k*T)/(p*molecular_area*np.sqrt(2))
    print('l = ', l)
    rho_air = 1.225
    mu = (1/3)*l*c*M*g_to_kg*mc

    print('viscosity = ', mu*10**7, 'microPoises')
    print('moles per cubic meter = ', mc)

    D = (1/3)*l*c
    print('diffusive flux D = ', D, 'm/s')

    Cv = (3/2)*R
    Kt = (1/3)*l*c*Cv*mc
    print('Kt = ', Kt, 'J/mKs')



def main():
    k = 1.36*10**-23
    R = 8.314
    T = np.array([273.15, 20+273.15, 600+273.15])
    p = 101325  #Pascals
    M = 28.9647 #g/mol
    mc = p/(R*T)
    calc_viscosity(M, k, R, T, p, mc)


if __name__ == '__main__':
    main()