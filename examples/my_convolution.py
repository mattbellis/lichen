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
        #val *= -math.erf((sigma_squared-tau*(mean+pt))/(np.sqrt(2)*sigma*tau))
        #val *=  math.erf((sigma_squared+tau*(mean-10.0))/(np.sqrt(2)*sigma*tau))
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
    #conv_means = [0.0,0.0,-1.0]
    #conv_sigmas = [0.1,1.0,0.5]
    conv_means = [0.0]
    conv_sigmas = [1.0]
    colors = ['r','g','b']
    for cm,cs,color in zip(conv_means,conv_sigmas,colors):

        #z,convpts = my_smear_with_gaussian_convolution(x,gpts,cm,cs)
        #fig1.add_subplot(1,3,1)
        #plt.plot(x,convpts,color=color)

        #fig1.add_subplot(1,3,2)
        #plt.plot(x,z,color=color)

        # Exponential
        #z,convpts = my_smear_with_gaussian_convolution(x,exp_pts,cm,cs)

        z_ana = my_smear_with_gaussian_convolution_analytic(x_exp,tau,cm,cs)

        fig1.add_subplot(1,3,3)
        #plt.plot(x_exp,z,color=color)
        plt.plot(x_exp,z_ana,color='c')
        #plt.set_xlim(0,5)

    # Need this command to display the figure.
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
