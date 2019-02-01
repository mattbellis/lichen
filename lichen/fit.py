from scipy import optimize
import numpy as np

import scipy.stats as stats

from scipy.optimize import fmin_bfgs,fmin_l_bfgs_b,approx_fprime

################################################################################
class Parameter:

    value = None
    limits = (None,None)

    def __init__(self):
        self.value = None
        self.limits = (None,None)

    def __init__(self,value,lo,hi):
        if lo>value:
            print("value {0} is lower than the lo limit {1}!".format(value,lo))
            exit()
        if hi<value:
            print("value {0} is greater than the hi limit {1}!".format(value,hi))
            exit()
        self.value = value
        self.limits = (lo,hi)

    def __init__(self,value,limits):
        if limits is None:
            self.value = value
            self.limits = limits
        else:
            if len(limits)==2:
                if limits[0]>value:
                    print("value {0} is lower than the lo limit {1}!".format(value,limits[0]))
                    exit()
                if limits[1]<value:
                    print("value {0} is greater than the hi limit {1}!".format(value,limits[1]))
                    exit()

                self.value = value
                self.limits = limits


################################################################################
def pretty_print_parameters(params_dictionary):
    print("Floating parameters")
    for key in params_dictionary:
        if key=="mapping":
            continue
        for k in params_dictionary[key].keys():
            if params_dictionary[key][k].limits is not None:
                print("{0:20} {1:20} {2}".format(key,k,params_dictionary[key][k].value))

    print("\nFixed parameters")
    for key in params_dictionary:
        if key=="mapping":
            continue
        for k in params_dictionary[key].keys():
            if params_dictionary[key][k].limits is None:
                print("{0:20} {1:20} {2}".format(key,k,params_dictionary[key][k].value))


################################################################################
def get_numbers(params_dictionary):
    numbers = []
    for key in params_dictionary:
        if key=="mapping":
            continue
        #print(params_dictionary[key])
        for k in params_dictionary[key].keys():
            if k=="number":
                numbers.append(params_dictionary[key][k].value)
    return numbers
################################################################################
################################################################################
def reset_parameters(params_dict,params):

    mapping = params_dict["mapping"]

    # We will only set the ones that are mapped onto the values (params)
    # that are floated by the algorithm
    for m in mapping:
        
        major_key = m[0]
        minor_key = m[1]
        value_idx = m[2]
        params_dict[major_key][minor_key].value = params[value_idx]

################################################################################

################################################################################
def pois(mu, k):
    ret = -mu + k*np.log(mu)
    return ret
################################################################################

####################################################
def errfunc(pars, x, y, fix_or_float=[],params_dictionary=None,pdf=None,verbose=False):
  ret = None

  ##############################################################################
  # Update the dictionary with the new parameter values
  reset_parameters(params_dictionary,pars)
  ##############################################################################

  nums = get_numbers(params_dictionary)
  ntot = sum(nums)
  ndata = len(x[0])

  # Assume the data (x) is actually multidimensional
  ret = (-np.log(pdf(params_dictionary, x, frange=(0,10))) .sum()) - pois(ntot, ndata)

  if verbose:
      print("NLL (errfunc): {0}     ntot: {1}    ndata:{2}".format(ret,ntot,ndata))

  return ret
################################################################################

################################################################################
def get_values_and_bounds(pars):
    values = []
    bounds = []
    mapping = []
    # Loop over the parameter names
    values_idx = 0
    for key in pars:

        print(key)
        
        for k in pars[key].keys():
            print("\t",k)
            bound = pars[key][k].limits
            # Check to see if the bounds are None. If they are
            # then this parameter is *fixed* in the fit. 
            if bound is not None:
                bounds.append(pars[key][k].limits)
                values.append(pars[key][k].value)
                mapping.append([key,k,values_idx])
                values_idx += 1

    pars['mapping'] = mapping

    return values,bounds

################################################################################
def fit_emlm(func, pars, data, verbose=False):

    # Need to do this for the fit.
    # Need to pull out starting values and
    p0,parbounds = get_values_and_bounds(pars)

    p1 = fmin_l_bfgs_b(errfunc, p0, args=(data, data, [], pars, func, verbose), bounds=parbounds, approx_grad=True, epsilon=1e-8)#, maxiter=100 )#,factr=0.1)

    finalvals = p1[0]

    reset_parameters(pars,finalvals)

    return p0,finalvals


'''
# Maybe for uncertainties, call something that calculates the Hessian?
retvals = fmin_bfgs(errfunc, p1[0], args=(data, data, [], pars,pdf), full_output=True,epsilon=1e-8)
print(p1[0])
print(retvals[0])
invh = retvals[3]
npars = len(p1[0])
print(npars)
for i in range(npars):
print(np.sqrt(invh[i][i]))
'''



