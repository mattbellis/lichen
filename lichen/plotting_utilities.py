import numpy as np
import matplotlib.pylab as plt
import scipy.integrate as integrate

################################################################################
# Plotting code for pdf
################################################################################
def plot_pdf(x,ypts,bin_width=1.0,scale=1.0,efficiency=1.0,axes=None,fmt='-',subranges=None,linewidth=1):

    y = None
    plot = None

    if axes==None:
        axes=plt.gca()

    if subranges!= None:
        totnorm = 0.0
        srnorms = []
        y = []
        plot = []
        for srx,sr in zip(srxs,subranges):
            sry = np.ones(len(srx))
            norm = integrate.simps(sry,x=srx)
            srnorms.append(norm)
            totnorm += norm

        for tot_sry,norm,srx,sr in zip(tot_srys,srnorms,srxs,subranges):
            sry = np.ones(len(srx))
            norm /= totnorm

            ypts = np.ones(len(srx))
            ytemp,plottemp = plot_pdf(srx,ypts,bin_width=bin_width,scale=norm*scale,fmt=fmt,axes=axes)
            y.append(ytemp)
            plot.append(plottemp)
            #tot_sry += y


    else:
        y = np.array(ypts)
        y *= efficiency

        # Normalize to 1.0
        normalization = integrate.simps(y,x=x)
        y /= normalization

        #print "exp int: ",integrate.simps(y,x=x)
        #y *= (scale*bin_width)*efficiency
        y *= (scale*bin_width)

        plot = axes.plot(x,y,fmt,linewidth=linewidth)
        #ytot += y
        #ax0.plot(x,ytot,'b',linewidth=3)

    return y,plot


