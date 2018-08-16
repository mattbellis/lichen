import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy import optimize

################################################################################
def hist_err(values,bins=100,range=None,fmt='o',color='blue',ecolor='black',markersize=2,axes=None,barsabove=False,capsize=0,linewidth=1,normed=False,weights=None,label=None,alpha=0.0):

    nentries_per_bin, bin_edges, patches = plt.hist(values,bins=bins,
            range=range,alpha=alpha,weights=weights) # Make histogram transparent.

    # Create an errorbar plot using the info from the histogram.
    bin_width = bin_edges[1] - bin_edges[0] # Assumes evenly spaced bins.
    xpts = bin_edges[0:-1] + bin_width/2.0 # Get the bin centers and leave off
                                           # the last point which is the high
                                           # side of a bin.

    ypts = 1.0*nentries_per_bin.copy()
    xpts_err = bin_width/2.0
    ypts_err = np.sqrt(nentries_per_bin) # Use np.sqrt to take square root
                                         # of an array. We'll assume Gaussian
                                         # errors here.
    if normed:
        ntot = float(sum(nentries_per_bin))
        ypts /= ntot
        ypts_err /= ntot


    # If no axes are passed in, use the current axes available to plt.
    if axes==None:
        axes=plt.gca()

    ret = axes.errorbar(xpts, ypts, xerr=xpts_err, yerr=ypts_err,fmt=fmt,
            color=color,ecolor=ecolor,markersize=markersize,barsabove=barsabove,capsize=capsize,
            linewidth=linewidth,label=label,alpha=1.0)

    if normed:
        axes.set_ylim(0,2.0*max(ypts))

    return ret,xpts,ypts,xpts_err,ypts_err

################################################################################
def hist_2D(xvals,yvals,xbins=10,ybins=10,xrange=None,yrange=None,origin='lower',cmap=plt.cm.coolwarm,axes=None,aspect='auto',log=False,weights=None,zlim=(None,None)):

    # If no ranges are passed in, use the min and max of the x- and y-vals.
    if xrange==None:
        xrange = (min(xvals),max(xvals))
    if yrange==None:
        yrange = (min(yvals),max(xvals))

    # Note I am switching the expected order of xvals and yvals, following the 
    # comment in the SciPy tutorial.
    # ``Please note that the histogram does not follow the Cartesian convention 
    # where x values are on the abcissa and y values on the ordinate axis. Rather, 
    # x is histogrammed along the first dimension of the array (vertical), and y 
    # along the second dimension of the array (horizontal). 
    # This ensures compatibility with histogramdd.
    #
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
    H,xedges,yedges = np.histogram2d(yvals,xvals,bins=[ybins,xbins],range=[yrange,xrange],weights=weights)
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

    if log is True:
        H = np.log10(H)

    # If no axes are passed in, use the current axes available to plt.
    if axes==None:
        axes=plt.gca()

    ret = axes.imshow(H,extent=extent,interpolation='nearest',origin=origin,cmap=cmap,axes=axes,aspect=aspect,vmin=zlim[0],vmax=zlim[1])
    #colorbar = plt.colorbar(cax=axes)

    return ret,xedges,yedges,H,extent


################################################################################
def fit(func,xdata,ydata,starting_vals=None,yerr=None):

    npars = len(starting_vals)

    fit_params, cov_mat = optimize.curve_fit(func, xdata, ydata, starting_vals, sigma=yerr)

    fit_params_errs = []
    for i in xrange(npars):
        fit_params_errs.append(np.sqrt(cov_mat[i][j]))

    return fit_params,fit_params_errs,cov_mat




