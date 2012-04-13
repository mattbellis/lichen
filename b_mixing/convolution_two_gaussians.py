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

def smear_with_gaussian_convolution(x,y,mean,sigma):

    npts = len(x)

    convolving_term = stats.norm(mean,sigma)
    convolving_pts = convolving_term.pdf(x)

    # Try adding another Gaussian to the.
    convolving_term_2 = stats.norm(2.0,5.0)

    for i,pt in enumerate(convolving_pts):
        convolving_pts[i] += convolving_term_2.pdf(x[i])

    convolving_pts[i] /= 2.0

    convolved_function = signal.convolve(y/y.sum(),convolving_pts)

    # Have to carve out the middle of the curve, because
    # the returned array has too many points in it. 
    znpts = len(convolved_function)
    begin = znpts/2 - npts/2
    end = znpts/2 + npts/2

    print "%d %d %d %d" % (npts,znpts,begin,end)

    return convolved_function[begin:end],convolving_pts

################################################################################
# main
################################################################################
def main():
    
    ############################################################################
    # Make a figure on which to plot stuff.
    # This would be the same as making a TCanvas object.
    ############################################################################
    fig1 = plt.figure(figsize=(12,4),dpi=100,facecolor='w',edgecolor='k')

    ############################################################################
    # Now divide of the figure as if we were making a bunch of TPad objects.
    # These are called ``subplots".
    #
    # Usage is XYZ: X=how many rows to divide.
    #               Y=how many columns to divide.
    #               Z=which plot to plot based on the first being '1'.
    # So '111' is just one plot on the main figure.
    ############################################################################
    #subplot = fig1.add_subplot(1,1,1)

    lo = -10
    hi =  10
    npts = 1000

    ############################################################################
    # Generate values drawn from a normal (Gaussian) distribution.
    ############################################################################
    mean = 0.0
    sigma = 0.5
    rv = stats.norm(mean,sigma)
    x = np.linspace(lo,hi,npts)
    #print x
    gpts = rv.pdf(x)
    fig1.add_subplot(1,3,2)
    plt.plot(x,gpts,color='k')
    norm_norm = gpts.sum()*(x[3]-x[2])
    print norm_norm

    ############################################################################
    # Generate values drawn from a negative exponential
    ############################################################################
    tau = 1.0/1.547
    x_exp = np.linspace(-10,10,npts)
    exp_pts = np.exp(np.abs(x_exp)*(-tau))
    exp_norm = exp_pts.sum()*(x_exp[3]-x_exp[2])
    print exp_norm
    exp_pts /= exp_norm
    fig1.add_subplot(1,3,3)
    plt.plot(x_exp,exp_pts,color='k')

    ############################################################################
    # Try the convolution
    ############################################################################
    conv_means = [0.0,2.0]
    conv_sigmas = [1.0,5.0]
    #conv_means = [0.0]
    #conv_sigmas = [1.0]
    colors = ['r','g']
    for cm,cs,color in zip(conv_means,conv_sigmas,colors):
        z,convpts = smear_with_gaussian_convolution(x,gpts,cm,cs)
        fig1.add_subplot(1,3,1)
        plt.plot(x,convpts,color=color)

        fig1.add_subplot(1,3,2)
        plt.plot(x,z,color=color)

        # Exponential
        z,convpts = smear_with_gaussian_convolution(x,exp_pts,cm,cs)

        fig1.add_subplot(1,3,3)
        plt.plot(x_exp,z,color=color)
        #plt.set_xlim(0,5)

    # Need this command to display the figure.
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
