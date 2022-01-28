# # lattice_01_basic
# #
# # A computer program to generate four realizations of
# # a 0,1 one dimensional lattice process
# # on an n dimenesional lattice and
# # to calculate and plot statistics
# #
# # Programed by J.L. Wilson, January, 2011,
# # with kudos to an earlier program by Katrina Koski
# #
# # This program generates an n-dimensional lattice and fills it multiple
# # realizations of randomly generated values of 0 or 1, each with a
# # probability of 0.50.  Currently four realizations are generated;
# # statistics (mean, variance, standard deviation, coefficient of variation)
# # for each individual realization are calculated and recorded in a file,
# # Statistics_Table. "Cumulative density" is calculated for each position
# # in the lattice which can be used to estimate the Representative
# # Elementary Length of the process. The realization result, histogram,
# # density is plotted for each realization. There are two different
# # versions of the density plots in order to provide different prespectives. The
# # Density is the same in both kinds of plots..
# #
# clear all
# close all
#
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# np.random.seed(123456789)

def monte_carlo_sim(markov_lattice, markov_no_p):
    n = markov_no_p # number of lattice points
    # assumes that probability of 0 or 1 at each point is 0.50
    #
    x = 100
    location = np.arange(1, n+1) # location index
    # print(location)
    # lattice = np.zeros((x, n)) # initialize array for 4 realizations
    # print('lattice = ', lattice)
    #
    fid = open('Statistics_Table_part2', 'w') # create output file
    fid.write('                Statistics Table \n ') # start output file
    fid.write('           for #{} lattice points \n'.format(n))
    fid.write('--------------------------------------------- \n ')
    fid.write('Realization   Mean   Variance  Std Dev    CV     density_mean     density_variance\n')
    #
    HF = plt.figure(1)
    HF.tight_layout()
    # plt.show()
    # HF=figure;  # open figure HF for histograms
    #
    avg = np.zeros(n)
    vari = np.zeros(n)
    stddev = np.zeros(n)
    CV = np.zeros(x)
    density = np.zeros((x, n))
    dens_avg = np.zeros(n)
    dens_vari = np.zeros(n)
    dens_stddev = np.zeros(n)
    d_dens_dm = np.zeros((x, n))
    ddens_dm_avg = np.zeros(n)

    # generate the random lattice here.  this could be used in replacement of line 62.
    # lattice = np.random.randint(2, size=(x, n))
    # for problemo 1.4
    lattice = markov_lattice
    ## for problemo 1.3
    # lattice = np.random.choice([0, 1], size=(x, n), p=[.7, .3])
    print('rows = ', len(lattice), 'cols = ', len(lattice[0]))

    for j in range(x):
        density[j, :] = np.cumsum(lattice[j, :]) / location  # calc. density function
        # d_dens_dm[j, :] = np.cumsum(lattice[j, :])*-1 / location**2
        # print('density for realization {} = {}'.format(j, density))
    for k in range(n):  # n lattice points, storing stats at each position
        # print('problem mean = ', np.mean(lattice[:, k])) # calc. mean
        avg[k] = np.mean(lattice[:, k])
        vari[k] = np.var(lattice[:, k]) # calc. variance
        stddev[k] = np.sqrt(vari[k]) # calc. standard deviation
        # # CV[k] = stddev[k]/avg[k] # calc. coef. of variation

        dens_vari[k] = np.var(density[:, k])
        dens_stddev[k] = np.sqrt(dens_vari[k])
        # ddens_dm_avg[k] = np.mean(d_dens_dm[:, k])

        dens_avg[k] = np.mean(density[:, k])
    sumofcumdiff = np.zeros(n)
    diff = np.zeros(n)
    diff_stddev = np.zeros(n)
    for m in range(n):
        # print(m)
        if m == n-1:
            break
        else:
            diff[m] = dens_avg[m+1] - dens_avg[m]
            diff_stddev[m] = np.abs(dens_stddev[m+1] - dens_stddev[m])
            # sumofcumdiff[m] = np.cumsum(np.abs(diff))
            # print(sumofcumdiff)

    range_obs = np.max(dens_avg) - np.min(dens_avg)
    print('range = ', range_obs)
    RELcutoff = 0.025*np.mean(dens_avg)
    # RELcutoff = 0.3*range_obs
    count = 0
    for i, di in enumerate(dens_stddev):
        if di <= RELcutoff:
            count += 1
            if count > .01*n:
                print('length point = ', i, 'and', 'standard deviation = ', di, 'SANITY CHECK!!! REL CUTOFF = ', RELcutoff)
                break
        else:
            count = 0

    grad = diff_stddev
    plt.figure(6)
    plt.plot(grad)
    for i, di in enumerate(grad):
        if di <= RELcutoff:
            print('GRADIENT length point = ', i, 'and', 'standard deviation = ', di, 'SANITY CHECK!!! REL CUTOFF = ', RELcutoff)
            break


    fid.close()
    # plt.show()
    # sys.exit('check stats')
    plotstyle = dict(markersize=2, linewidth=0.75)
    plt.figure(1)
    plt.plot(avg, '-o', **plotstyle)
    plt.xlim(0, n)
    plt.ylim(0, 1)
    plt.xlabel('averaging distance (length)')
    plt.ylabel('mean')
    plt.title('ensemble mean of processes')

    plt.figure(2)
    plt.plot(stddev, '-^', **plotstyle)
    plt.xlim(0, n)
    plt.ylim(0, 0.75)
    plt.xlabel('averaging distance (length)')
    plt.ylabel('standard deviation')
    plt.title('ensemble standard deviation of processes')

    plt.figure(3)
    plt.plot(dens_avg, '-', **plotstyle)
    plt.xlabel('averaging distance (length)')
    plt.ylabel('mean')
    plt.title('ensemble mean of densities')

    plt.figure(4)
    plt.plot(dens_stddev, '-^', **plotstyle)
    plt.xlabel('averaging distance (length)')
    plt.ylabel('standard deviation')
    plt.title('ensemble standard deviation of densities')

    plt.show()

def markov_chain():
    # markovchain.m marfichrope
    #
    # a script file that
    # simulates a 2 member Markov chain
    # on a one dimensional lattice {1, 2, 3, ..., np}.
    # where np is the number of lattice points,
    # given an initial distribution [p1 p2] and a transition matrix P.
    #
    # John L. Wilson
    # Hydrology 508, New Mexico Tech,
    # January, 2011
    #
    # The program assumes that the states are labeled 1, 2
    #
    #
    # Initiate
    #
    P = [(.7, .3),
         (.2, .8)]
    # P=[[.7 .3], [.2 .8]]  # transition matrix
    p1 = P[0][0]
    p2 = P[0][1]
    # p1=P(1,1) p2=P(1,2) # probabilities for first lattice location
    no_p = 8000  # number of lattice (time or space) points
    location = np.arange(1, no_p + 1)  # location of lattice points
    # lattice = np.ones(1, np) # initial values at lattice points set to 1
    lattice = np.ones(no_p).astype(int)
    #
    # Compute
    #

    lattice[0] = math.floor(np.random.random() + p2)  # generate value 1 or 2 at first location
    # lattice[1] = (lattice[1]).astype(int)
    print('lattice1 = {}, {}'.format(lattice[1], type(lattice[1])))
    for j in np.arange(1, no_p, 1):
        # print('j = {}'.format(j))
        lattice[j] = math.floor(np.random.random() + P[lattice[j - 1]][1])  # generate other values
        # print('lattice at {} = {}'.format(j, lattice[j]))
    #
    #
    # Calculate experimental transition matrix, Pexp,
    # and compare to the model transition matrix, P,
    # to test the method. It takes about np=10,000
    # points to provide good test. Display on command line.
    #
    p11 = 0
    p12 = 0
    p21 = 0
    p22 = 0  # initiate counters
    n1 = 0
    n2 = 0
    nerror = 0
    #
    # The following comment line is an independent 0,1 Beroulli trial
    # process to generate a transition matrix = [0.5 0.5 0.5 0.5]
    # uncomment in order to test the transition matrix calculation.
    #
    # lattice= 1 + np.round(np.random.random(1, no_p)) # generate uncorrelated 1,2 processes
    # lattice = np.random.randint(2, size=no_p)

    # lattice[33] = 9
    # lattice[45] = 0
    # test values uncomment to test the error ck

    for j in np.arange(1, no_p, 1):
        # check total numbers of 1s and 2s check for incorrect values
        if lattice[j] == 0:
            n1 = n1 + 1
        elif lattice[j] == 1:
            n2 = n2 + 1
        else:
            nerror = nerror + 1

    print('n1 = ', n1)
    print('n2 = ', n2)
    print('nerror = ', nerror)

    number_ones = np.count_nonzero(lattice)
    print('number of ones = {}'.format(number_ones))
    number_zeros = lattice.shape[0] - number_ones
    print('number of zeros = {}'.format(number_zeros))

    if nerror != 0:
        print('------------------------')  # print errors to screen
        print('------------------------')
        print('Error in lattice values:')
        print('number of lattice points = ', no_p)
        print('number of ones = ', n1)
        print('number of twos = ', n2)
        print('number of incorrect lattice points = ', nerror)

    #
    for j in np.arange(1, no_p, 1):  # count up number of transition types
        if lattice[j] == lattice[j - 1]:
            if lattice[j] == 0:
                p11 = p11 + 1
            elif lattice[j] == 1:
                p22 = p22 + 1
        elif lattice[j] > lattice[j - 1]:
            p12 = p12 + 1
        elif lattice[j] < lattice[j - 1]:
            p21 = p21 + 1
    else:
        print('marissa added this in')
    print('p11 = ', p11)
    print('p22 = ', p22)
    print('p12 = ', p12)
    print('p21 = ', p21)
    #
    p11norm = p11 / (p11 + p12)
    p12norm = p12 / (p11 + p12)  # normalize transition counts
    p21norm = p21 / (p21 + p22)
    p22norm = p22 / (p21 + p22)  # since probs sum to one

    Pexp = ([p11norm, p12norm], [p21norm, p22norm])  # create exp. trans. matrix

    print('------------------------')  # print to screen
    print('------------------------')
    print('Number of lattice points = ', no_p)
    print('Experimental Transition Matrix: ')
    print('Pexp = {}'.format(Pexp))
    print('Target Transition Matrix:')
    print('P = {}'.format(P))
    print('------------------------')
    #
    #
    # Output
    #
    # plt.plot(location, lattice, '.')
    # plt.plot(lattice, '.')
    # if nerror == 0:
    #     plt.xlim(1, no_p)
    #     plt.ylim(-1, 2)
    # plt.title('binary markov chain')
    # plt.xlabel('lattice points')
    # plt.ylabel('value')

    # plt.show()

    return np.array(lattice), no_p

def main():
    markov_lattices = np.zeros((100, 8000))
    for i in range(10):
        markov_lattice, no_p = markov_chain()
        markov_lattices[i] = markov_lattice
        # print(markov_lattice, no_p)
    print('shape of multi markov lattice = ', markov_lattices.shape)
    monte_carlo_sim(markov_lattices, no_p)


if __name__ == '__main__':
    main()