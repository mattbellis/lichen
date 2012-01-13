#!/usr/bin/env python

################################################################################
# Make multiple figures (TCanvas) and subdivide them into mutiple 
# subplots (TPad). 
# 
# This is supposed to demonstrate a replacement for:
#
# TCanvas can();
# can.Divide(2,2);
#
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
    nfigs = 2
    figs = []
    for i in xrange(nfigs):
        figs.append(plt.figure(figsize=(8,6),dpi=100,facecolor='w',edgecolor='k'))

    ############################################################################
    # Now divide each of the figures as if we were making a bunch of 
    # TPad objects.
    #
    # These are called ``subplots".
    #
    # Usage is XYZ: X=how many rows to divide.
    #               Y=how many columns to divide.
    #               Z=which plot to plot based on the first being '1'.
    # So '111' is just one plot on the main figure.
    ############################################################################
    subplots = []
    for i in xrange(nfigs):
        # We'll make this a nfigs x nsubplots_per_fig to store the subplots
        subplots.append([])
        for j in xrange(1,5):
            # Divide the figure into a 2x2 grid.
            subplots[i].append(figs[i].add_subplot(2,2,j))

        # Adjust the spacing between the subplots, to allow for better 
        # readability.
        figs[i].subplots_adjust(wspace=0.4,hspace=0.4)

    ############################################################################
    # Generate values drawn from normal (Gaussian) distributions.
    ############################################################################
    mean = [4.0, 5.0, 6.0, 7.0]
    sigma = [1.0, 2.0]
    values = []
    for i,s in enumerate(sigma):
        values.append([])
        for m in mean:
            values[i].append(np.random.normal(m,s,10000))

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
    colors = ['green','red','blue','purple']
    alphas = [0.90,0.50]
    h = []
    for i,s in enumerate(sigma):
        for j,m in enumerate(mean):
            h = subplots[i][j].hist(values[i][j],bins=100,range=(0.0,10.0),facecolor=colors[j],alpha=alphas[i],
            histtype='stepfilled')

    ############################################################################
    # Let's format these histograms.
    # Note that we will do this by changing the values on the subplot (Axes), 
    # not the histogram object.
    ############################################################################

    for i,s in enumerate(sigma):
        for j,m in enumerate(mean):
            subplots[i][j].set_xlim(-1,11)
            subplots[i][j].set_ylim(0,500)
            
            subplots[i][j].set_xlabel('x variable')
            subplots[i][j].set_ylabel('# events')
            
            # Note that we can easily include Latex code
            text = r"$\mathrm{Gaussian:}\ \mu=%2.1f,\ \sigma=%2.1f$" % (m,s)
            subplots[i][j].set_title(text)


    # Need this command to display the figure.
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
