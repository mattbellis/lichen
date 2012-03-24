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

import lichen.lichen as lch
import lichen

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
    ############################################################################
    h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(values,bins=50,range=(0.0,10.0),color='pink') # 

    ############################################################################
    # Let's format this histogram. 
    # Note that we will do this by changing the values on the subplot (Axes), 
    # not the histogram object.
    ############################################################################
    subplot.set_xlim(-1,11)
    
    subplot.set_xlabel('x variable',fontsize=20)
    subplot.set_ylabel('# events',fontsize=20)
    
    # Note that we can easily include Latex code
    subplot.set_title(r'$\mathrm{Gaussian\ distribution:}\ \mu=5,\ \sigma=1$',fontsize=30)

    # Set the number of tick marks on the x-axis.
    subplot.locator_params(nbins=8)

    fit_params,fit_params_errs,cov_mat = lch.fit(lichen.pdf_gaussian(),xpts,ypts,[100.0,5.0,1.0],yerr=ypts_err)

    # Need this command to display the figure.
    plt.show()

################################################################################
# Top-level script evironment
################################################################################
if __name__ == "__main__":
    main()
