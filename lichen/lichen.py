import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

################################################################################
def hist_err(values,bins=100,range=None,fmt='o',color='blue',ecolor='black'):

    nentries_per_bin, bin_edges, patches = plt.hist(values,bins=bins,
            range=range,alpha=0.0) # Make histogram transparent.

    # Create an errorbar plot using the info from the histogram.
    bin_width = bin_edges[1] - bin_edges[0] # Assumes evenly spaced bins.
    xpts = bin_edges[0:-1] + bin_width/2.0 # Get the bin centers and leave off
                                           # the last point which is the high
                                           # side of a bin.

    ypts = nentries_per_bin
    xpts_err = bin_width/2.0
    ypts_err = np.sqrt(nentries_per_bin) # Use np.sqrt to take square root
                                         # of an array. We'll assume Gaussian
                                         # errors here.

    ret = plt.errorbar(xpts, ypts, xerr=xpts_err, yerr=ypts_err,fmt=fmt,
            color=color,ecolor=ecolor)

    return ret,xpts,ypts,xpts_err,ypts_err


################################################################################
def fit(func,xdata,ydata,starting_vals=None,yerr=None):

    npars = len(starting_vals)

    fit_params, cov_mat = optimize.curve_fit(func, xdata, ydata, starting_vals, sigma=yerr)

    fit_params_errs = []
    for i in xrange(npars):
        fit_params_errs.append(np.sqrt(cov_mat[i][j]))

    return fit_params,fit_params_errs,cov_mat




