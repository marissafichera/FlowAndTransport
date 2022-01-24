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

# np.random.seed(123456789)

n = 5000 # number of lattice points
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

# generate the random lattice here.  this could be used in replacement of line 62.
lattice = np.random.randint(2, size=(x, n))
print('rows = ', len(lattice), 'cols = ', len(lattice[0]))

for j in range(x):
    density[j, :] = np.cumsum(lattice[j, :]) / location  # calc. density function
    # print('density for realization {} = {}'.format(j, density))
for k in range(n):  # n lattice points, storing stats at each position
    print(k)
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
    # avg[j] = np.mean(lattice[j, :]) # calc. mean
    # vari[j] = np.var(lattice[j, :]) # calc. variance
    # stddev[j] = np.sqrt(vari[j]) # calc. standard deviation
    # CV[j] = stddev[j]/avg[j] # calc. coef. of variation

    # print('lattice problem = ', lattice[:, k])
    # print('lattice shape = ', lattice[:, k].mean())

    # print('problem mean = ', np.mean(lattice[:, k])) # calc. mean
    avg[k] = np.mean(lattice[:, k])
    vari[k] = np.var(lattice[:, k]) # calc. variance
    stddev[k] = np.sqrt(vari[k]) # calc. standard deviation
    # # CV[k] = stddev[k]/avg[k] # calc. coef. of variation
    dens_avg[k] = np.mean(density[:, k])
    dens_vari[k] = np.var(density[:, k])
    dens_stddev[k] = np.sqrt(dens_vari[k])



    # fid.write('     {}          {:0.2f}     {:0.2f}     {:0.2f}     {:0.2f}    {:0.2f}    {:0.2f}\n'.format(j, avg[j], vari[j], stddev[j], CV[j], dens_avg[j], dens_vari[j]))
        # j,average(j),variance(j),std_dev(j),CV(j)); # print statistics
    #
    # print('k: {}'.format(k))
    # print('avg at position {}: {}'.format(k, avg[k]))
    # print('dens_avg: {}'.format(dens_avg[j]))
    # print('dens_stddev: {}'.format(dens_stddev[j]))

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
plt.plot(dens_avg, '-o', **plotstyle)
plt.xlabel('averaging distance (length)')
plt.ylabel('mean')
plt.title('ensemble mean of densities')

plt.figure(4)
plt.plot(dens_stddev, '-^', **plotstyle)
# plt.xlim(0, n)
# plt.ylim(0, 1)
plt.xlabel('averaging distance (length)')
plt.ylabel('standard deviation')
plt.title('ensemble standard deviation of densities')

plt.show()

# plot the density as function of averaging interval (one plot for all)

# DFa= plt.figure(3) # open new figure DF for density plot
# DFa.tight_layout()
# plotstyle = dict(markersize=2, linewidth=0.75)
# for j in range(x):
#     plt.plot(dens_avg[j],'-o', label='{}'.format(j+1), **plotstyle)
#     plt.plot(dens_vari[j], '-^', label='{}'.format(j+1), **plotstyle)
#     plt.ylim(0, 1)
#     plt.xlabel('averaging interval')
#     plt.ylabel('density')
#     plt.title('Density')
# #
# #
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