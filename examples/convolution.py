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

import scipy.stats as stats
import scipy.signal as signal

################################################################################
# main
################################################################################
def main():
    
    ############################################################################
    # Make a figure on which to plot stuff.
    # This would be the same as making a TCanvas object.
    ############################################################################
    fig1 = plt.figure(figsize=(8,6),dpi=100,facecolor='w',edgecolor='k')

    ############################################################################
    # Now divide of the figure as if we were making a bunch of TPad objects.
    # These are called ``subplots".
    #
    # Usage is XYZ: X=how many rows to divide.
    #               Y=how many columns to divide.
    #               Z=which plot to plot based on the first being '1'.
    # So '111' is just one plot on the main figure.
    ############################################################################
    subplot = fig1.add_subplot(1,1,1)

    ############################################################################
    # Generate values drawn from a normal (Gaussian) distribution.
    ############################################################################
    mean = 1.0
    sigma = 0.2
    rv = stats.norm(mean,sigma)
    x = np.linspace(-np.minimum(rv.dist.b, 5), np.minimum(rv.dist.b, 5),100)
    print x
    h = plt.plot(x, rv.pdf(x))

    subplot.set_xlim(-10,10)

    ############################################################################
    # Try the convolution
    ############################################################################
    conv_mean = 0.0
    conv_sigma = 1.2
    conv_rv = stats.norm(conv_mean,conv_sigma)
    y = np.linspace(-np.minimum(conv_rv.dist.b, 5), np.minimum(conv_rv.dist.b, 5),100)
    print y
    h = plt.plot(x, conv_rv.pdf(y))

    z = signal.convolve(x,y)
    print y

    # Need this command to display the figure.
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
