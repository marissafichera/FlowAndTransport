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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


# np.random.seed(123456789)

n = 5000 # number of lattice points
x = 100 # number of realizations
# assumes that probability of 0 or 1 at each point is 0.50
#
location = np.arange(1, n+1) # location index
# print(location)
lattice = np.zeros((x, n)) # initialize array for x realizations
# print('lattice = ', lattice)
#
fid = open('Statistics_Table', 'w') # create output file
fid.write('                Statistics Table \n ') # start output file
fid.write('           for #{} lattice points \n'.format(n))
fid.write('--------------------------------------------- \n ')
fid.write('Realization   Mean   Variance  Std Dev    CV\n')
#
HF = plt.figure(1)
HF.tight_layout()
# plt.show()
# HF=figure;  # open figure HF for histograms
#
avg = np.zeros(x)
vari = np.zeros(x)
stddev = np.zeros(x)
CV = np.zeros(x)
density = np.zeros((x, n))

# generate the random lattice here.  this could be used in replacement of line 62.
rlattice = np.random.randint(2, size=(x, n))
lattice = np.random.randint(2, size=(x, n))
avg_process = np.zeros((x, n))
vari_process = np.zeros((x, n))
density_process = np.zeros((x, n))
# print('size avg process1 = {}'.format(avg_process.size))
for j in range(x):  # x realizations
    # lattice = np.random.randint(2, size=(x, n))
    for k in range(n):
        avg_process[j, k] = np.mean(lattice[j, 0:k])
        vari_process[j, k] = np.var(lattice[j, 0:k])
        # print(location[k])
        # print('lattice = ', lattice[j, 0:k])
        # print('cumsum = ', np.cumsum(lattice[j, 0:k]))
        # density_process[j, k] = np.cumsum(lattice[j, 0:k])/(k+1)

        # print('j = {}'.format(j), 'k = {}'.format(k))
        # print('avg at {}, {} = {}'.format(j, k, avg_process[j, k]))
        # print('variance at {}, {} = {}'.format(j, k, vari_process[j, k]))
        # print('density at {}, {} = {}'.format(j, k, density_process[j, k]))
    # print(avg_process)
    # print('size = ', avg_process.size)
    # plt.plot(avg_process[j], 'o')
    # plt.show()
    # sys.exit('check avg values')
    density_process[j, :] = np.cumsum(lattice[j, :]) / location
    # plt.plot(density_process[j, :])
    # plt.show()

sys.exit('as;lkdfj;askldjf;sklj')

    # lattice[j, :] = np.round((np.random.random(n))) # generate four 0,1 processes
                                     # revise this line to change probs
    # print('lattice {}:'.format(j), lattice[j,:])

    # HF.subplot(2,2,j) # histogram for one of x realizations
    # plt.subplot(2, 2, j+1)
    # plt.hist((lattice[j,:]), align='right') # plot histogram for real j
    # labels, counts = np.unique(lattice[j, :], return_counts=True)
    # plt.bar(labels, counts, align='center')
    # plt.xlabel('value')
    # plt.ylabel('number')
    # plt.xlim(-1, 2)
    # plt.ylim(0, n)
    # plt.gca().set_xticks([0, 1])

    # h = findobj(gca,'Type','patch');
    # set(h,'FaceColor','r','EdgeColor','w'); #surround bar with white
    # set(gca,'XTick',[0.25 0.75],'XTickLabel',char('0', '1'));...
    #     # center values
    #

#     avg[j] = np.mean(lattice[j, :]) # calc. mean
#     vari[j] = np.var(lattice[j, :]) # calc. variance
#     stddev[j] = np.sqrt(vari[j]) # calc. standard deviation
#     # print('STANDARD DEV {} = '.format(j), stddev)
#     CV[j] = stddev[j]/avg[j] # calc. coef. of variation
#     fid.write('     {}          {:0.2f}     {:0.2f}     {:0.2f}     {:0.2f}\n'.format(j, avg[j], vari[j], stddev[j], CV[j]))
#         # j,average(j),variance(j),std_dev(j),CV(j)); # print statistics
#     #
#     density[j, :] = np.cumsum(lattice[j, :])/location # calc. density function
#
#         # mavg = np.mean(lattice[j, k])
#
# fid.close()
# plt.show()
#
# # plot the density as function of averaging interval (one plot for all)
#
# DFa= plt.figure(3) # open new figure DF for density plot
# DFa.tight_layout()
# plotstyle = dict(markersize=2, linewidth=0.75)
# for j in range(x):
#     plt.plot(density[j],'-o', label='{}'.format(j+1), **plotstyle)
#     plt.ylim(0, 1)
#     plt.xlabel('averaging interval')
#     plt.ylabel('density')
#     plt.title('Density')
#
#
#     target = 0.5*np.ones(n) # add the expected value to the density plot
#     plt.plot(location, target, '--')
# plt.legend()
# plt.show()
# #
# # plot the density as function of averaging interval (separate plots)
# DFb= plt.figure(4) # open new figure RP for plots of realizations
# DFb.tight_layout()
# for j in range(x):  # plot the x density funcitons
#     plt.subplot(2,2,j+1) # density for each of x realizations
#     plt.plot(density[j, :], '-o', **plotstyle) # plot
#     plt.xlabel('averaging interval')
#     plt.ylabel('density')
#     plt.xlim(0, n)
#     plt.ylim(0, 1)
#     plt.plot(location, target, '--')
# plt.show()
# # plot the realizations
# RP = plt.figure(5)  # open new figure RP for plots of realizations
# RP.tight_layout()
# for j in range(x):  # plot the x realizations
#     plt.subplot(3, 2, j+1) # histogram for each of x realizations
#     plt.bar(location, lattice[j, :]) # plot the 0,1 realization
#     print(lattice[j])
#     plt.xlabel('event')
#     plt.ylabel('value')
#     plt.xlim(0, n)
#     plt.ylim(0, 1.2)
# plt.show()
