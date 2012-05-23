import convolving_functions as conv
import numpy as np

x = np.array([1.0])

xerr = np.array([2.0])
tau = 1.0/1.54
pdf = conv.convolve_exp_with_gaussians_per_event_errors(x,xerr,tau)[0]
print "xerr: %f\ttau: %f\tpdf: %f" % (xerr[0],tau,pdf)
tau = 1.0/1.64
pdf = conv.convolve_exp_with_gaussians_per_event_errors(x,xerr,tau)[0]
print "xerr: %f\ttau: %f\tpdf: %f" % (xerr[0],tau,pdf)

xerr = np.array([0.1])
tau = 1.0/1.54
pdf = conv.convolve_exp_with_gaussians_per_event_errors(x,xerr,tau)[0]
print "xerr: %f\ttau: %f\tpdf: %f" % (xerr[0],tau,pdf)
tau = 1.0/1.64
pdf = conv.convolve_exp_with_gaussians_per_event_errors(x,xerr,tau)[0]
print "xerr: %f\ttau: %f\tpdf: %f" % (xerr[0],tau,pdf)
