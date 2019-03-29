import scipy.stats as stats
import numpy as np
import matplotlib.pylab as plt

import lichen as lch

np.random.seed(0)

mean = [1.0,2.0]
cov = [[0.2*0.2, 0.0],
       [0.0, 0.1*0.1]]

data = np.random.multivariate_normal(mean,cov,1000)
data_x = data.transpose()[0]
data_y = data.transpose()[1]

idx = (data_y<2.0) | (data_y>2.1)
x = data_x[idx]
y = data_y[idx]

print(len(x))

plt.figure()
plt.subplot(2,2,1)
plt.hist(x,bins=50,range=(0,2))

xpts = np.linspace(0,2,100)
pdf = stats.norm(1.0,0.2)
plt.plot(xpts,pdf.pdf(xpts)*len(x)*((2-0)/50))


plt.subplot(2,2,2)
plt.hist(y,bins=50)

plt.subplot(2,2,4)
lch.hist2d(x,y,xbins=50,ybins=50,xrange=(-1,3),yrange=(0,3.5))


plt.show()


