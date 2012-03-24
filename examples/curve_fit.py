# Routine linear fit demo.
# Adapted from http://physics.nyu.edu/pine/pymanual/graphics/graphics.html
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import scipy.optimize

################################################################################
# Define function to calculate reduced chi-squared
################################################################################
def red_chi_sq(func, x, y, dy, params):
    resids = y - func(x, *params)
    chisq = ((resids/dy)**2).sum()
    return chisq/float(x.size-params.size)

################################################################################
# Define fitting function for linear fit
################################################################################
def linear(x,m,b):
    return (m*x) + b

################################################################################
# main 
################################################################################
def main():

    #infile = open('linear_fit_data.dat')

    # Create some arrays to store the data.
    xdata = np.array([])
    ydata = np.array([])
    yerr  = np.array([])

    #xdata = np.array([1.1, 2.0, 3.1, 4.2, 4.9])
    #ydata = np.array([1.0, 2.5, 3.0, 4.0, 5.1])
    #yerr  = np.array([0.4, 0.2, 0.4, 0.5, 0.4])
    xdata = np.linspace(0,4,10)
    ydata = linear(xdata,1.1,0.4) + 0.2*np.random.normal(size=len(xdata))
    yerr  = np.linspace(0.2,0.2,len(xdata))
    print xdata
    #for x,y,e in zip(xdata,ydata,yerr):
        #print "%5.2f %5.2f %5.2f" % (x,y,e)

    # Perform linear fit using Levenburg-Marquardt algorithm
    # Take some guesses at what the values are 
    m = 1.1 # slope
    b = 0.4 # y-intercept
    nlfit, nlpcov = scipy.optimize.curve_fit(linear, xdata, ydata, p0=[m,b], sigma=yerr)

    # Unpack output and give outputs of fit nice names
    m_fit, b_fit = nlfit           # returned values of fitting parameters
    dm_fit = np.sqrt(nlpcov[0][0]) # uncertainty in 1st fitting parameter
    db_fit = np.sqrt(nlpcov[1][1]) # uncertainty in 2nd fitting parameter

    # Compute reduced chi square.
    # Need to do this to get proper uncertainties on the fit values.
    rchisq = red_chi_sq(linear, xdata, ydata, yerr, nlfit)

    # Create fitted data for plotting the fit function over the measured data.
    qfit = np.array([xdata.min(), xdata.max()])
    Sfit = linear(qfit, m_fit, b_fit)

    # Plot data and fit
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(qfit, Sfit)

    ax1.errorbar(xdata,ydata,yerr, fmt="or", ecolor="black")

    ax1.set_xlabel("x-axis")
    ax1.set_ylabel("y-axis")

    ax1.text(0.5, 0.83,
                 "slope $= {0:0.3f} \pm {1:0.3f}$".format(m_fit, dm_fit),
                      ha="right", va="bottom", transform = ax1.transAxes)
    ax1.text(0.5, 0.90,
            "intercept $= {0:0.3f} \pm {1:0.3f}$".format(b_fit, db_fit),
                 ha="right", va="bottom", transform = ax1.transAxes)

    plt.savefig("fitted_data.png")
    plt.show()

################################################################################
if __name__ == "__main__":
    main()
