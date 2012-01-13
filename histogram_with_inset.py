#!/usr/bin/env python

################################################################################
# Make a histogram and plot it on a figure (TCanvas).
# 
# Make a second histogram and plot it in an inset.
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
    subplot_main = fig1.add_subplot(1,1,1)

    # Create another subplot that occupies the upper right of the figure.
    # Divide the figure up into a 3x3 grid and use the upper right.
    subplot_inset = fig1.add_subplot(3,3,3)

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
    # The returned object is a tuple of nentries, bins, patches.
    h_main = subplot_main.hist(values,bins=100,range=(0.0,10.0),facecolor='green',alpha=0.50,
            histtype='stepfilled',axes=subplot_main)

    h_inset = subplot_inset.hist(values,bins=100,range=(6.0,10.0),facecolor='red',alpha=0.50,
            histtype='stepfilled',axes=subplot_inset)

    ############################################################################
    # Let's format these histograms.
    # Note that we will do this by changing the values on the subplot (Axes), 
    # not the histogram object.
    ############################################################################

    #################### MAIN ##################################################
    subplot_main.set_xlim(-1,11)
    subplot_main.set_ylim(0,500)
    
    subplot_main.set_xlabel('x variable',fontsize=20)
    subplot_main.set_ylabel('# events',fontsize=20)
    
    # Note that we can easily include Latex code
    subplot_main.set_title(r'$\mathrm{Gaussian\ distribution:}\ \mu=5,\ \sigma=1$',fontsize=30)

    #################### INSET #################################################
    subplot_inset.set_xlabel('x variable',fontsize=10)
    # For formatting, turn off the last tickmark ([-1] in a list).
    # Comment these out to see why I do this. 
    subplot_inset.yaxis.get_major_ticks()[-1].label1On = False

    # More fine grained tick formatting for the inset. 
    # Get the list of what the tick marks are, by default, and keep only 
    # every other tick. 
    tick_locs = subplot_inset.xaxis.get_ticklocs()
    new_tick_locs = []
    for i,t in enumerate(tick_locs):
        if i%2==0:
            new_tick_locs.append(t)
    # Reload the new list of ticks, except the last one.
    subplot_inset.xaxis.set_ticks(new_tick_locs[0:-1])
    

    # Need this command to display the figure.
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
