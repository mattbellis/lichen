#!/usr/bin/env python

# Adapted from the Glowing python example.
# http://glowingpython.blogspot.com/2012/02/convolution-with-numpy.html

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

################################################################################
# A wrapper to scipy.signal.convolve
################################################################################
def convolve_func_with_gaussians(x,y,mean=[0.0],sigma=[1.0],fractions=[1.0]):

    # x is x-values of function.
    # y is y-values of function.

    npts = len(x)
    ngaussians = len(mean)

    convolving_pts = np.zeros(npts)

    for i in range(0,ngaussians):

        convolving_term = stats.norm(mean[i],sigma[i])
        convolving_pts += fractions[i]*convolving_term.pdf(x)

    # Normalize
    convolving_pts /= sum(fractions)

    convolved_function = signal.convolve(y/y.sum(),convolving_pts)

    # Have to carve out the middle of the curve, because
    # the returned array has too many points in it. 
    znpts = len(convolved_function)
    begin = znpts/2 - npts/2
    end = znpts/2 + npts/2

    #print "%d %d %d %d" % (npts,znpts,begin,end)

    return convolved_function[begin:end],convolving_pts

################################################################################
# Numerical convolution.
################################################################################
def convolve_exp_with_gaussians_numerical(x,tau,mean=[0.0],sigma=[1.0],fractions=[1.0],window=3.0,nwindow=500):

    # This is for the window around each point which we use to 
    # calculate the convolving points.
    window *= max(sigma)
    # nwindow is the number of Gaussians to use in the smearing.
    # Should this be a function of the window itself?

    y = np.array([])

    for pt in x:

        # Get the x points used for the convolution.
        temp_pts = np.linspace(pt-window,pt+window,nwindow)

        val = 0.0
        for m,s,f in zip(mean,sigma,fractions):
            convolving_term = stats.norm(m,s)
            val += (np.exp(-abs(temp_pts)*tau)*f*convolving_term.pdf(pt-temp_pts)).sum() #/np.exp(-abs(temp_pts)*tau).sum()

        #y = np.append(y,val/sum(fractions))
        y = np.append(y,val)

    normalization = 1.0
    if len(x)>3:
        normalization = y.sum()*(x[3]-x[2])
    #print "norm: %f" % (normalization)

    return y/normalization

################################################################################

################################################################################
# Numerical convolution with per-event errors.
################################################################################
def convolve_exp_with_gaussians_per_event_errors(x,xerr,tau,mean=[0.0],sigma=[1.0],fractions=[1.0],window=3.0,nwindow=500):

    # This is for the window around each point which we use to 
    # calculate the convolving points.
    window *= max(sigma)
    # nwindow is the number of Gaussians to use in the smearing.
    # Should this be a function of the window itself?

    y = np.array([])

    for pt,err in zip(x,xerr):

        # Get the x points used for the convolution.
        temp_pts = np.linspace(pt-window,pt+window,nwindow)

        val = 0.0
        for m,s,f in zip(mean,sigma,fractions):
            m *= err
            s *= err
            convolving_term = stats.norm(m,s)
            val += (np.exp(-abs(temp_pts)*tau)*f*convolving_term.pdf(pt-temp_pts)).sum() #/np.exp(-abs(temp_pts)*tau).sum()
            #print "val: ",val

        #y = np.append(y,val/sum(fractions))
        y = np.append(y,val)

    normalization = 1.0
    if len(x)>3:
        normalization = y.sum()*(x[3]-x[2])
    #print "norm: %f" % (normalization)

    return y/normalization

################################################################################



################################################################################
# Numerical convolution with per-event errors.
################################################################################
def convolve_b_mixing_with_gaussians_per_event_errors(x,xerr,tau,mean=[0.0],sigma=[1.0],fractions=[1.0],window=3.0,nwindow=500):

    # This is for the window around each point which we use to 
    # calculate the convolving points.
    window *= max(sigma)
    # nwindow is the number of Gaussians to use in the smearing.
    # Should this be a function of the window itself?

    y = np.array([])

    for pt,err in zip(x,xerr):

        # Get the x points used for the convolution.
        temp_pts = np.linspace(pt-window,pt+window,nwindow)

        val = 0.0
        for m,s,f in zip(mean,sigma,fractions):
            m *= err
            s *= err
            convolving_term = stats.norm(m,s)
            #val += (pdfs.pdf_bmixing(deltat,[gamma,p_over_q,deltaM,deltaG,q1,q2])*tau)*f*convolving_term.pdf(pt-temp_pts)).sum() #/np.exp(-abs(temp_pts)*tau).sum()
            #print "val: ",val

        #y = np.append(y,val/sum(fractions))
        y = np.append(y,val)

    normalization = 1.0
    if len(x)>3:
        normalization = y.sum()*(x[3]-x[2])
    #print "norm: %f" % (normalization)

    return y/normalization

################################################################################



