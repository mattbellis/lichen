#!/usr/bin/env python

################################################################################
# Define a data container
################################################################################

data = []

################################################################################
# Import the standard libraries in the accepted fashion.
################################################################################
from math import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import lichen.lichen as lch
import lichen.pdfs as pdfs

import minuit

################################################################################
# main
################################################################################
def main():
    
    ############################################################################
    # Make a figure on which to plot stuff.
    # This would be the same as making a TCanvas object.
    ############################################################################
    figs = []
    subplots = []
    for i in xrange(1):
        figs.append(plt.figure(figsize=(8,6),dpi=100,facecolor='w',edgecolor='k'))
        subplots.append([])
        for j in range(0,4):
            subplots[i].append(figs[i].add_subplot(2,2,j+1))

    ############################################################################
    # Generate values drawn from a normal (Gaussian) distribution.
    ############################################################################
    deltat_min = -20
    deltat_max =  20
    deltat_range =  deltat_max-deltat_min

    deltat_mc = []
    for i in range(0,4):
        deltat_mc.append(deltat_range*np.random.rand(10000) + deltat_min)

    print "deltat MC: %d" % (len(deltat_mc))

    deltat = np.linspace(deltat_min,deltat_max,1000)

    gamma = 1.0/1.547
    p_over_q = 1.01
    A = 1.0
    deltaM = 0.4
    deltaG = 0.0

    charges = [[+1,+1], [-1,-1], [+1,-1], [-1,+1]]

    maxes = []

    for i in range(0,4):

        q1 = charges[i][0]
        q2 = charges[i][1]
        N = pdfs.pdf_bmixing(deltat,[gamma,p_over_q,deltaM,deltaG,q1,q2])

        maxes.append(max(N))

        subplots[0][i].plot(deltat,N,'-',markersize=2)

    max_prob = max(maxes)

    '''
    events = [np.array([]),np.array([]),np.array([]),np.array([])]
    n=0
    nevents = 100
    print "Generating %d events." % (nevents)
    while n<nevents:

        if n%1000==0:
            print n

        for i in range(0,4):

            q1 = charges[i][0]
            q2 = charges[i][1]

            val = deltat_range*np.random.rand()+deltat_min

            prob = pdfs.pdf_bmixing(val,[gamma,p_over_q,deltaM,deltaG,q1,q2])

            test = max_prob*np.random.rand()
            
            if test<prob:
                events[i] = np.append(events[i],val)
                n += 1

    for i in range(0,4):
        #subplots[1][i].hist(events[i],bins=50)
        figs[1].add_subplot(2,2,i+1)
        lch.hist_err(events[i],bins=50)
        subplots[1][i].set_xlim(deltat_min,deltat_max)
        subplots[1][i].set_ylim(0)
        #subplots[1][i].set_ylim(0,nevents/16)

    Npp = len(events[0])
    Nmm = len(events[1])
    Npm = len(events[2])
    Nmp = len(events[3])

    print "%d %d %d %d" % (Npp,Nmm,Npm,Nmp)

    Acp = (Npp-Nmm)/float(Npp+Nmm) 
    deltaAcp = np.sqrt((sqrt(Npp)/Npp)**2 + (sqrt(Nmm)/Nmm)**2)

    print "Acp: %f +/- %f" % (Acp,deltaAcp)

    Acp = 2*(1-np.abs(1.0/p_over_q))

    print "Acp: %f" % (Acp)
    '''

    '''
    # Fit function.
    gamma = 1.0/1.547
    p_over_q = 1.01
    A = 1.0
    deltaM = 0.4
    deltaG = 0.0
    #N = pdfs.pdf_bmixing(deltat,[gamma,p_over_q,deltaM,deltaG,q1,q2])
    n0 = Npp
    n1 = Nmm
    n2 = Npm 
    n3 = Nmp
    p0 = [gamma,p_over_q,deltaM,deltaG,n0,n1,n2,n3]
    print p0
    #p1 = sp.optimize.fmin(pdfs.extended_maximum_likelihood_function,p0,args=(events,deltat_mc), maxiter=10000, maxfun=10000)

    #print p1

    #data = [events,deltat_mc]
    #m = minuit.Minuit(pdfs.extended_maximum_likelihood_function_minuit,p=p0)
    #print m.values
    #m.migrad()
    '''

    # Need this command to display the figure.
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
