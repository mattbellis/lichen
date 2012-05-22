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
def convolve_func_with_gaussian(x,y,mean=0.0,sigma=1.0):

    # x is x-values of function.
    # y is y-values of function.

    npts = len(x)

    convolving_term = stats.norm(mean,sigma)
    convolving_pts = convolving_term.pdf(x)

    convolved_function = signal.convolve(y/y.sum(),convolving_pts)

    # Have to carve out the middle of the curve, because
    # the returned array has too many points in it. 
    znpts = len(convolved_function)
    begin = znpts/2 - npts/2
    end = znpts/2 + npts/2

    #print "%d %d %d %d" % (npts,znpts,begin,end)

    return convolved_function[begin:end],convolving_pts

################################################################################
# A wrapper to scipy.signal.convolve
################################################################################
def convolve_func_with_two_gaussian(x,y,mean=[0.0],sigma=[1.0],fractions=[1.0]):

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





























################################################################################
def smear_with_gaussian_convolution(x,y,mean,sigma):

    npts = len(x)

    convolving_term = stats.norm(mean,sigma)
    convolving_pts = convolving_term.pdf(x)

    '''
    # Try adding another Gaussian to the.
    convolving_term_2 = stats.norm(2.0,5.0)

    for i,pt in enumerate(convolving_pts):
        convolving_pts[i] += convolving_term_2.pdf(x[i])
    '''

    convolved_function = signal.convolve(y/y.sum(),convolving_pts)

    # Have to carve out the middle of the curve, because
    # the returned array has too many points in it. 
    znpts = len(convolved_function)
    begin = znpts/2 - npts/2
    end = znpts/2 + npts/2

    print "%d %d %d %d" % (npts,znpts,begin,end)

    return convolved_function[begin:end],convolving_pts


################################################################################
def my_smear_with_gaussian_convolution(x,y,mean,sigma):

    npts = len(x)

    convolving_term = stats.norm(mean,sigma)
    convolving_pts = convolving_term.pdf(x)

    convolved_function = np.array([])

    for i in range(0,npts):
        convolved_function = np.append(convolved_function,0.0)
        for j in range(0,npts,10):

            convolved_function[i] += y[j]*convolving_term.pdf(x[i]-x[j])

            j+=0.5


    # Have to carve out the middle of the curve, because
    # the returned array has too many points in it. 
    znpts = len(convolved_function)
    begin = znpts/2 - npts/2
    end = znpts/2 + npts/2

    print "%d %d %d %d" % (npts,znpts,begin,end)

    normalization =  convolved_function.sum()*(x[3]-x[2])
    
    ret =  convolved_function[begin:end]/normalization

    #print convolved_function[begin:end]
    return ret,convolving_pts

################################################################################
def my_smear_with_gaussian_convolution_analytic(x,tau,mean,sigma):

    npts = len(x)

    tau = 1.0/tau

    y = np.array([])

    sigma_squared = sigma*sigma
    tau_squared = tau*tau
    tau_sigma = tau*sigma
    const0 = np.sqrt(np.pi/2.0)*sigma
    const1 = np.exp((sigma_squared-2*mean*tau)/(2*tau_squared))
    const2 = (np.sqrt(2)*sigma*tau)

    for pt in x:
        abspt = np.abs(pt)
        #abspt = pt
        #val = const0*const1
        #val *= math.erf((sigma_squared+(tau*(pt-mean))/(np.sqrt(2)*tau_sigma)))
        val = const0
        val *= np.exp((sigma_squared+2*tau*(mean-abspt))/(2*tau_squared))
        #val *= np.exp((sigma_squared-2*tau*(mean+abspt))/(2*tau_squared))
        #val *= -np.exp(-abspt/tau)
        #print (math.erf((mean-pt-10)/(np.sqrt(2)*sigma)) - math.erf((mean-pt+10)/(np.sqrt(2)*sigma)))
        #val *= (math.erf((mean-pt-10)/(np.sqrt(2)*sigma)) - math.erf((mean-pt+10)/(np.sqrt(2)*sigma)))
        #val *= (math.erf(mean/(np.sqrt(2)*sigma)) - math.erf((pt+mean)/(np.sqrt(2)*sigma)))
        #val *= -(1 - math.erf((mean-abspt)/(np.sqrt(2)*sigma)))
        '''
        val *= ((math.erf((sigma_squared+(tau*mean)))/const2) -
               (math.erf((sigma_squared+(tau*(mean-abspt)))/const2)) )
        '''
        '''
        if pt>0:
            val *=  math.erfc((sigma_squared-tau*(mean+pt))/(np.sqrt(2)*sigma*tau))
            #val *= -math.erf((sigma_squared+tau*(mean+pt))/(np.sqrt(2)*sigma*tau))
        else:
            val *=  math.erfc((sigma_squared-tau*(mean+pt))/(np.sqrt(2)*sigma*tau))
            #val *= math.erf((sigma_squared+tau*(mean+pt))/(np.sqrt(2)*sigma*tau))
        '''
        '''
        if pt>0:
            val *= (math.erf((mean-pt)/(np.sqrt(2)*sigma)) - 1)
        else:
            val *= -(1+math.erf((mean-pt)/(np.sqrt(2)*sigma)))
        '''

        print "pt: %f %f %f %f" % (pt,const0,const1,val)

        y = np.append(y,val)

    #print convolved_function[begin:end]
    return y


################################################################################
def my_smear_with_gaussian_convolution_numerical(x,tau,mean,sigma):

    npts = len(x)

    #tau = 1.0/tau

    y = np.array([])

    window = 3*sigma
    nwindow = 500

    for pt in x:

        temp_pts = np.linspace(pt-window,pt+window,nwindow)
        convolving_term = stats.norm(mean,sigma)
        val = (np.exp(-abs(temp_pts)*tau)*convolving_term.pdf(pt-temp_pts)).sum()
        #val = np.exp(-np.abs(pt)*tau)

        y = np.append(y,val)


    normalization =  y.sum()*(x[3]-x[2])

    return y/normalization

################################################################################
def my_smear_with_gaussian_convolution_numerical_per_event_errors(x,tau,mean,sigma):


    npts = len(x)

    #tau = 1.0/tau

    y = np.array([])

    nwindow = 100

    for pt,sig in zip(x,sigma):

        window = 4*sig
        #print sig

        temp_pts = np.linspace(pt-window,pt+window,nwindow)
        convolving_term = stats.norm(mean,sig)
        val = (np.exp(-abs(temp_pts)*tau)*convolving_term.pdf(pt-temp_pts)).sum() #/np.exp(-abs(temp_pts)*tau).sum()

        #val = np.exp(-np.abs(pt)*tau)

        #print "%f %f" % (sig,val)

        y = np.append(y,val)


    normalization = 1.0
    if len(x)>3:
        normalization = y.sum()*(x[3]-x[2])
    print "norm: %f" % (normalization)

    return y/normalization


################################################################################
def my_smear_with_gaussian_convolution_numerical_per_event_errors_pdf(x,tau,mean,sigma):


    y = np.array([])

    nwindow = 100

    pt = x
    sig = sigma

    window = 4*sig
    #print sig

    temp_pts = np.linspace(pt-window,pt+window,nwindow)
    convolving_term = stats.norm(mean,sig)
    val = (np.exp(-abs(temp_pts)*tau)*convolving_term.pdf(pt-temp_pts)).sum() #/np.exp(-abs(temp_pts)*tau).sum()
    #print convolving_term.pdf(pt-temp_pts)
    #print val

    #val = np.exp(-np.abs(pt)*tau)
    #print "%f %f" % (sig,val)

    y = np.append(y,val)

    normalization = 1.0
    #print "norm: %f" % (normalization)

    return y/normalization

################################################################################
def triple_gaussian_convolution_per_event_errors_pdf(x,tau,mean,sigma):


    y = np.array([])

    nwindow = 400

    pt = x
    sig = sigma

    window = 32*sig
    #print sig

    temp_pts = np.linspace(pt-window,pt+window,nwindow)
    temp_pts2 = pt - np.linspace(pt-window,pt+window,nwindow)
    
    g0 = stats.norm(mean,1.0*sig)
    g1 = stats.norm(mean,2.0*sig)
    g2 = stats.norm(mean,8.0*sig)

    val = (np.exp(-abs(temp_pts)*tau)*(g0.pdf(temp_pts2)+0.10*g1.pdf(temp_pts2)+0.20*g2.pdf(temp_pts2))).sum() 
    
    #print val

    #val = np.exp(-np.abs(pt)*tau)
    #print "%f %f" % (sig,val)

    y = np.append(y,val)

    normalization = 1.0
    #print "norm: %f" % (normalization)

    return y/normalization

