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

################################################################################
# Linear function
################################################################################
def pdf_bmixing(deltat,pars):
    
    gamma = pars[0]
    p_over_q = pars[1]
    deltaM = pars[2]
    deltaG = pars[3]
    q1 = pars[4]
    q2 = pars[5]

    qq = q1*q2

    A =  (1/2.0)*(1+qq)*(p_over_q**(2*q1)) + (1/2.0)*(1-qq) # coefficient of cosh term
    B = -(1/2.0)*(1+qq)*(p_over_q**(2*q1)) + (1/2.0)*(1-qq) # coefficient of cos  term
    C = 0.0 # coefficient of sinh term
    D = 0.0 # coefficient of sin  term

    #print "%d %f %f %f %f" % (qq,A,B,C,D)
    
    N = (1.0/4.0)*np.exp(-gamma*np.abs(deltat))*(
            A*np.cosh(deltaG*deltat/2)-B*np.cos(deltaM*deltat) +
            C*np.sinh(deltaG*deltat/2)-D*np.sin(deltaM*deltat) )

    return N

