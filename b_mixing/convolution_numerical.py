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

################################################################################
# A wrapper to scipy.signal.convolve
################################################################################
def smear_with_gaussian_convolution(x,y,mean,sigma):

    npts = len(x)

    convolving_term = stats.norm(mean,sigma)
    convolving_pts = convolving_term.pdf(x)

    convolved_function = signal.convolve(y/y.sum(),convolving_pts)

    # Have to carve out the middle of the curve, because
    # the returned array has too many points in it. 
    znpts = len(convolved_function)
    begin = znpts/2 - npts/2
    end = znpts/2 + npts/2

    print "%d %d %d %d" % (npts,znpts,begin,end)

    return convolved_function[begin:end],convolving_pts

################################################################################
# Numerical convolution.
################################################################################
def smear_exponential_with_gaussian_convolution_numerical(x,tau,mean,sigma,window=3.0,nwindow=500):

    # This is for the window around each point which we use to 
    # calculate the convolving points.
    window *= sigma
    # nwindow is the number of Gaussians to use in the smearing.
    # Should this be a function of the window itself?

    y = np.array([])

    for pt in x:

        # Get the x points used for the convolution.
        temp_pts = np.linspace(pt-window,pt+window,nwindow)

        convolving_term = stats.norm(mean,sigma)

        val = (np.exp(-abs(temp_pts)*tau)*convolving_term.pdf(pt-temp_pts)).sum()
        #val = np.exp(-np.abs(pt)*tau)

        y = np.append(y,val)


    normalization =  y.sum()*(x[3]-x[2])

    return y/normalization


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

    ############################################################################
    # The range of the plotting functions.
    ############################################################################
    lo = -10
    hi =  10
    npts = 1000

    ############################################################################
    # Generate values drawn from a negative exponential
    ############################################################################
    x = np.linspace(lo,hi,npts)
    x_exp = np.linspace(lo,hi,npts)

    tau = 1.0/1.547

    exp_pts = np.exp(np.abs(x_exp)*(-tau))
    # Normalize the exponential.
    exp_norm = exp_pts.sum()*(x_exp[3]-x_exp[2])
    print exp_norm
    exp_pts /= exp_norm

    # Print the exponentials.
    subplots[1] = fig1.add_subplot(1,3,2)
    subplots[1].set_title('Double exponential')
    porg0 = subplots[1].plot(x_exp,exp_pts,color='k',label='original')

    subplots[2] = fig1.add_subplot(1,3,3)
    subplots[2].set_title('Double exponential')
    porg1 = subplots[2].plot(x_exp,exp_pts,color='k',label='original')

    ############################################################################
    # Try the convolution
    ############################################################################
    #conv_means = [0.0,0.0,-1.0]
    #conv_sigmas = [0.1,1.0,0.5]
    conv_means = [0.0]
    conv_sigmas = [1.0]
    colors = ['r','g','b']
    for cm,cs,color in zip(conv_means,conv_sigmas,colors):

        # Smear a double exponential with Gaussians.
        z,convpts = smear_with_gaussian_convolution(x,exp_pts,cm,cs)

        subplots[0] = fig1.add_subplot(1,3,1)
        subplots[0].set_title('Convolving function')
        subplots[0].plot(x,convpts,color=color)

        fig1.add_subplot(1,3,2)
        pconv0 = subplots[1].plot(x_exp,z,color=color,label='convolved')
        subplots[1].set_ylim(0,0.45)
        subplots[1].legend(loc=0)

        # Exponential
        z_num = smear_exponential_with_gaussian_convolution_numerical(x_exp,tau,cm,cs)

        fig1.add_subplot(1,3,3)
        pconv1 = subplots[2].plot(x_exp,z_num,color='g',label='convolved (numerical)')
        subplots[2].set_ylim(0,0.45)
        subplots[2].legend(loc=0)

    ############################################################################
    # For comparing the scipy and my numerical calculation
    ############################################################################
    print z-z_num

    # Need this command to display the figure.
    fig1.subplots_adjust(left=0.05, bottom=None, right=0.95, wspace=0.2, hspace=None)
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
