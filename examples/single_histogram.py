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
    mean = 5.0
    sigma = 1.0
    values = np.random.normal(mean,sigma,10000)

    ############################################################################
    # Histogram of the data.
    # Note that we histogram by calling a member function of the subplot.
    #
    # The return values in this example are:
    #
    # nentries_per_bin: The number of entries in each bin of the historam.
    #                   This is just a list of integers.
    #
    # bins_edges: This is a list of floats that contains the low-edge of the 
    #             bins, plus one more entry for the high-edge of the 
    #             highest bin.
    #
    # patches: This is a list of the graphics objects (I think) that make up 
    #          the histogram.
    #       
    ############################################################################
    nentries_per_bin, bin_edges, patches = subplot.hist(values,bins=100,
            range=(0.0,10.0),facecolor='green',alpha=0.75,histtype='stepfilled')

    # Alternatively, one could just create the histogram object, and worry about 
    # associating it with a subplot at a later time.
    '''
    h = plt.hist(values,bins=10,range=(0.0,10.0),facecolor='green',alpha=0.75,
            histtype='stepfilled')
    '''

    ############################################################################
    # Let's format this histogram. 
    # Note that we will do this by changing the values on the subplot (Axes), 
    # not the histogram object.
    ############################################################################
    subplot.set_xlim(-1,11)
    subplot.set_ylim(0,500)
    
    subplot.set_xlabel('x variable',fontsize=20)
    subplot.set_ylabel('# events',fontsize=20)
    
    # Note that we can easily include Latex code
    subplot.set_title(r'$\mathrm{Gaussian\ distribution:}\ \mu=5,\ \sigma=1$',fontsize=30)

    # Need this command to display the figure.
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
