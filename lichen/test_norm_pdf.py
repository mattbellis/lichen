import numpy as np
import matplotlib,pylab as plt

import scipy as sp
import scipy.stats as stats
import scipy.integrate as integrate

import pdfs as pdfs 

#rv = stats.expon(scale=5)
#
x = np.linspace(0.1,20,1000)
#print(rv.pdf(x))

#plt.figure()
#plt.plot(x,rv.pdf(x))

#normalization = integrate.simps(rv.pdf(x),x=x)
#print(normalization)
#
#xnorm = np.linspace(0.5,0.7,1000)
#normalization = integrate.simps(rv.pdf(xnorm),x=xnorm)
#print(normalization)

#ynorm = rv.pdf(xnorm)/normalization

# nnf: numerically normalize function
'''
def nnf(func,normrange=(0,10),data=None):

    xnorm = np.linspace(normrange[0],normrange[1],1000)

    normalization = integrate.simps(rv.pdf(xnorm),x=xnorm)
    print(normalization)

    return rv.pdf(data)/normalization
'''

#n = nnf(rv,(0.5,0.7))
#print(n)


#xnorm = np.linspace(0.5,0.7,1000)
#normalization = integrate.simps(rv.pdf(xnorm)/n,x=xnorm)
#print(normalization)

################################################################################
rv = stats.lognorm(s=0.5,loc=1,scale=3.0)
x = np.linspace(0.1,20,1000)
plt.figure()
plt.plot(x,rv.pdf(x))

#n = nnf(rv,(5,6))
#print(n)

data = rv.rvs(size=10000)
plt.figure()
plt.hist(data,bins=100)

subset = data[(data>5)*(data<7)]
pdfdata,xnorm = pdfs.nnf(rv,normrange=(5,7),data=subset)

rv2 = stats.lognorm(s=1.6,loc=2,scale=4.0)
pdfdata2,xnorm = pdfs.nnf(rv2,normrange=(5,7),data=subset)

print(np.log(pdfdata).sum())
print(np.log(pdfdata2).sum())

rv3 = stats.lognorm(s=1.6,loc=2,scale=4.0)
pdfdata3,xnorm = pdfs.nnf(rv2,normrange=(5,7))
plt.figure()
plt.plot(xnorm,pdfdata3)

pdfdata4,xnorm = pdfs.nnf(rv,normrange=(5,7))
plt.plot(xnorm,pdfdata4)

'''
pdfdata = nnf(rv,normrange=(0.1,20),data=data)
print(pdfdata)
plt.figure()
plt.plot(data,pdfdata,'.')

pdfdata = nnf(rv,normrange=(5,6),data=data)
print(pdfdata)
plt.figure()
plt.plot(data,pdfdata,'.')
'''

plt.show()


