#!/usr/bin/env python

# Adapted from the Glowing python example.
# http://glowingpython.blogspot.com/2012/02/convolution-with-numpy.html

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

import scipy.stats as stats
import scipy.signal as signal

from convolving_functions import *

################################################################################
# main
################################################################################
def main():
    
    ############################################################################
    # Make a figure on which to plot stuff.
    # This would be the same as making a TCanvas object.
    ############################################################################
    fig1 = plt.figure(figsize=(12,4),dpi=100,facecolor='w',edgecolor='k')
    subplots = [None,None,None]

    subplots[0] = fig1.add_subplot(1,3,1)
    subplots[0].set_title('Convolving function')

    subplots[1] = fig1.add_subplot(1,3,2)
    subplots[1].set_title('Errors')

    subplots[2] = fig1.add_subplot(1,3,3)
    subplots[2].set_title('Double exponential')

    # Second figure
    fig2 = plt.figure(figsize=(12,4),dpi=100,facecolor='w',edgecolor='k')
    subplots2 = [None,None,None]

    subplots2[0] = fig2.add_subplot(1,3,1)
    subplots2[0].set_title('My numerical')

    subplots2[1] = fig2.add_subplot(1,3,2)
    subplots2[1].set_title('Difference')

    subplots2[2] = fig2.add_subplot(1,3,3)
    subplots2[2].set_title('Fractional difference')

    ############################################################################
    # The range of the plotting functions.
    ############################################################################
    lo = -20
    hi =  20
    npts = 1000

    x = np.linspace(lo,hi,npts)

    xerr = np.random.normal(1.0,0.25,npts)
    #xerr = 0.5*np.random.random(npts) + 1

    ############################################################################
    # Generate values drawn from a negative exponential
    # We will convolve this with a Gaussian distribution.
    ############################################################################
    tau = 1.0/1.547
    yexp = np.exp(np.abs(x)*(-tau))

    # Make sure exponential is normalized
    exp_norm = yexp.sum()*(x[3]-x[2])
    yexp /= exp_norm
    #print exp_norm

    herr = subplots[1].hist(xerr,color='r',label='errors',bins=50)

    porg1 = subplots[2].plot(x,yexp,color='k',label='original')
    #porg1_2 = subplots2[2].plot(x,yexp,color='k',label='original')

    ############################################################################
    # Convolve with either 1 or 2 Gaussians (3 different convolutions)
    ############################################################################
    #conv_means = [[0.0],[0.0,0.0],[-1.0,0.0]]
    #conv_sigmas = [[1.0],[1.0,5.0],[1.0,0.5]]
    #conv_fractions = [[1.0],[1.0,1.0],[1.0,0.5]]
    #colors = ['r','g','b']
    #linestyle = ['--','-.',':']

    conv_means = [[0.0],[0.0,0.0]]
    conv_sigmas = [[1.0],[1.0,5.0]]
    conv_fractions = [[1.0],[1.0,1.0]]
    colors = ['r','g']
    linestyle = ['--','-.']

    for cm,cs,frac,color,lstyle in zip(conv_means,conv_sigmas,conv_fractions,colors,linestyle):

        # Numerical using my method.
        znum = convolve_exp_with_gaussians_per_event_errors(x,xerr,tau,cm,cs,frac,4.0,500)
        #print znum
        #exit()

        pconv1 = subplots[2].plot(x,znum,color=color,label='convolved',linestyle=lstyle)

        pconv1num = subplots2[0].plot(x,znum,color=color,label='convolved',linestyle=lstyle)

        subplots[2].set_ylim(0,0.45)
        subplots[2].legend(loc=0)

        subplots2[0].set_ylim(0,0.30)
        subplots2[0].legend(loc=0)

    # Need this command to display the figure.
    fig1.subplots_adjust(left=0.05, bottom=None, right=0.95, wspace=0.2, hspace=None)
    fig2.subplots_adjust(left=0.05, bottom=None, right=0.95, wspace=0.2, hspace=None)
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
