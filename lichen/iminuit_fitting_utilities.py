import numpy as np

################################################################################
# Convert dictionary to kwd arguments
################################################################################
def dict2kwd(d,verbose=False):

    keys,vals = d.keys(),d.values()

    params_names = ()

    kwd = {}
    for k,v in d.items():
        if verbose:
            print(k,v)
        params_names += (k,)
        kwd[k] = v['start_val']
        if 'fix' in v and v['fix']==True:
            new_key = "fix_%s" % (k)
            kwd[new_key] = True
        if 'limits' in v:
            new_key = "limit_%s" % (k)
            kwd[new_key] = v['limits']
        if 'error' in v:
            new_key = "error_%s" % (k)
            kwd[new_key] = v['error']

    ''' 
    if 'num_bkg' in keys:
        print "YES!",d['num_bkg']
    '''

    return params_names,kwd

################################################################################
################################################################################
# Helper function
################################################################################
class Struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

################################################################################
# Helper fitting function
################################################################################
class Minuit_FCN:
    def __init__(self,data,params,function_to_minimize):
        self.data = data
        self.function_to_minimize = function_to_minimize

        ################
        params_names = ()
        #limits = {}
        kwd = {}
        for k,v in params.items():
            params_names += (k,)
            '''
            if 'var_' in k and 'limits' in v:
                new_key = "%s_limits" % (k)
                limits[new_key] = v['limits']
            '''
        ################

        self.params = params_names
        self.params_dict = params
        
        varnames = params_names

        self.func_code = Struct(co_argcount=len(params),co_varnames=varnames)
        self.func_defaults = None # Optional but makes vectorize happy

        print("Finished with __init__")

    def __call__(self,*arg):
        
        data0 = self.data[0]

        #val = emlf_normalized_minuit(data0,arg,self.func_code.co_varnames,self.params_dict)
        val = self.function_to_minimize(data0,arg,self.func_code.co_varnames,self.params_dict)

        return val

################################################################################

################################################################################
# Poisson function
################################################################################
def pois(mu, k):
    # mu = # of data returned by the fit
    # k  = # of data events
    ret = -mu + k*np.log(mu)
    return ret
################################################################################


