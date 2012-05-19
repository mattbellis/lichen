#!/usr/bin/env python

# Adapted from the Glowing python example.
# http://glowingpython.blogspot.com/2012/02/convolution-with-numpy.html

################################################################################
# Make a histogram and plot it on a figure (TCanvas).
################################################################################

################################################################################
# Import the standard libraries in the accepted fashion.
################################################################################
import math
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
    
    np.random.seed(10)

    ############################################################################
    # Make a figure on which to plot stuff.
    # This would be the same as making a TCanvas object.
    ############################################################################
    fig1 = plt.figure(figsize=(12,4),dpi=100,facecolor='w',edgecolor='k')

    lo = -10
    hi =  10
    npts = 10000

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
    #conv_means = [0.0,0.0,-1.0]
    #conv_sigmas = [0.1,1.0,0.5]
    conv_means = [0.0]
    #conv_sigmas = [1.0]
    #conv_sigmas = [0.8*np.random.rand(npts)+0.2]
    conv_sigmas = [0.00001*np.random.rand(npts)+1.05]
    #colors = ['r','g','b']
    colors = ['r']
    for cm,cs,color in zip(conv_means,conv_sigmas,colors):

        #z,convpts = my_smear_with_gaussian_convolution(x,gpts,cm,cs)
        #fig1.add_subplot(1,3,1)
        #plt.plot(x,convpts,color=color)

        #fig1.add_subplot(1,3,2)
        #plt.plot(x,z,color=color)

        # Exponential
        #z,convpts = my_smear_with_gaussian_convolution(x,exp_pts,cm,cs)

        #z_ana = my_smear_with_gaussian_convolution_analytic(x_exp,tau,cm,cs)
        #z_ana = my_smear_with_gaussian_convolution_numerical(x_exp,tau,cm,cs)
        z_ana = my_smear_with_gaussian_convolution_numerical_per_event_errors(x_exp,tau,cm,cs)

        fig1.add_subplot(1,3,3)
        #plt.plot(x_exp,z,color=color,linewidth=5)
        plt.plot(x_exp,z_ana,color='c')
        #plt.set_xlim(0,5)

        #fig1.add_subplot(1,3,1)
        #plt.plot(x_exp,cs,'o',color='c',markersize=1)

    # Generate some events.
    print "Generating some events......"
    events = np.array([])
    errs = np.array([])
    nevents = 10000
    zero = np.array([0.0])
    #max_val = my_smear_with_gaussian_convolution_numerical_per_event_errors_pdf(0.0,tau,0.0,1.0)
    max_val = triple_gaussian_convolution_per_event_errors_pdf(0.0,tau,0.0,1.0)
    #max_val = np.exp(0.0)
    print "max_val: %f" % (max_val)
    #exit(-1)
    i=0
    while i<nevents:

        if (i%100==0):
            print i
        

        x = 20*np.random.rand() - 10.0
        cs = 0.1*np.random.rand() + 1.0
        
        prob = triple_gaussian_convolution_per_event_errors_pdf(x,tau,0.0,cs)

        test_val = max_val*np.random.rand()

        if test_val<prob:
            events = np.append(events,x)
            errs = np.append(errs,cs)
            i += 1

    fig2 = plt.figure(figsize=(12,4),dpi=100,facecolor='w',edgecolor='k')
    fig2.add_subplot(1,2,1)
    plt.hist(events,bins=50)
    fig2.add_subplot(1,2,2)
    plt.hist(errs,bins=50)
    # Need this command to display the figure.
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
