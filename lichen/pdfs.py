import numpy as np

################################################################################
# Linear function
################################################################################
def pdf_linear():
    ret = lambda p, x: p[0]+p[1]*x
    return ret

################################################################################
# Gaussian (normal function)
################################################################################
def pdf_gaussian():
    ret = lambda p, x: (p[0]/p[2])*np.exp(-((x - p[1])**2)/(2.0*p[2]*p[2]))
    return ret

################################################################################
# Adding two PDFs
################################################################################
def pdf_addition(lambda_func_0,lambda_func_1):
    #norm = np.sqrt(p[0]*p[0]+p[1]*p[1])
    ret = lambda p, x: (p[0]**2/np.sqrt(p[0]*p[0]+p[1]*p[1]))*lambda_func_0(p[2:4],x) + (p[1]**2/np.sqrt(p[0]*p[0]+p[1]*p[1]))*lambda_func_1(p[4:7],x)
    return ret

################################################################################
# Chi^2 minimization function.
################################################################################
def chi2_function(fit_function):
    ret = lambda p, x, y: fit_function(p, x)-y
    return ret

################################################################################
# Define function to calculate reduced chi-squared
################################################################################
def red_chi_sq(func, x, y, dy, params):
    resids = y - func(x, *params)
    chisq = ((resids/dy)**2).sum()
    return chisq/float(x.size-params.size)
