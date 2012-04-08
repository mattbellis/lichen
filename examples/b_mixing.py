#!/usr/bin/env python

################################################################################
# Make a histogram and plot it on a figure (TCanvas).
################################################################################

################################################################################
# Import the standard libraries in the accepted fashion.
################################################################################
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import lichen.lichen as lch
import lichen.pdfs as pdfs

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
    for i in xrange(2):
        figs.append(plt.figure(figsize=(8,6),dpi=100,facecolor='w',edgecolor='k'))
        subplots.append([])
        for j in range(0,4):
            subplots[i].append(figs[i].add_subplot(2,2,j+1))

    ############################################################################
    # Generate values drawn from a normal (Gaussian) distribution.
    ############################################################################
    deltat = np.linspace(-10,10,1000)

    gamma = 1.0/1.547
    p_over_q = 1.5
    A = 1.0
    deltaM = 0.4
    deltaG = 0.0

    charges = [[+1,+1],
               [-1,-1],
               [+1,-1],
               [-1,+1]]

    maxes = []

    for i in range(0,4):

        q1 = charges[i][0]
        q2 = charges[i][1]
        N = pdfs.pdf_bmixing(deltat,[gamma,p_over_q,deltaM,deltaG,q1,q2])

        maxes.append(max(N))

        subplots[0][i].plot(deltat,N,'-',markersize=2)

    max_prob = max(maxes)

    events = [[],[],[],[]]
    n=0
    nevents = 10000
    while n<nevents:

        for i in range(0,4):

            q1 = charges[i][0]
            q2 = charges[i][1]

            val = 20*np.random.rand()-10

            prob = pdfs.pdf_bmixing(val,[gamma,p_over_q,deltaM,deltaG,q1,q2])

            test = max_prob*np.random.rand()
            
            if test<prob:
                events[i].append(val)
                n += 1

    for i in range(0,4):
        subplots[1][i].hist(events[i],bins=50)
        #subplots[1][i].set_xlim(-10,10)
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
    ############################################################################
    # Histogram of the data.
    ############################################################################
    h = lch.hist_err(values,bins=50,range=(0.0,10.0),color='pink') # 

    ############################################################################
    # Let's format this histogram. 
    # Note that we will do this by changing the values on the subplot (Axes), 
    # not the histogram object.
    ############################################################################
    subplot.set_xlim(-1,11)
    
    subplot.set_xlabel('x variable',fontsize=20)
    subplot.set_ylabel('# events',fontsize=20)
    
    # Note that we can easily include Latex code
    subplot.set_title(r'$\mathrm{Gaussian\ distribution:}\ \mu=5,\ \sigma=1$',fontsize=30)

    # Set the number of tick marks on the x-axis.
    subplot.locator_params(nbins=8)
    '''

    # Need this command to display the figure.
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
