import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

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
no_p = 200  # number of lattice (time or space) points
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
plt.plot(lattice, '.')
if nerror == 0:
    plt.xlim(1, no_p)
    plt.ylim(-1, 2)
plt.title('binary markov chain')
plt.xlabel('lattice points')
plt.ylabel('value')

plt.show()
