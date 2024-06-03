import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import curve_fit
from scipy.special import erf

def exp_gaus_single(x, A, center, sigma, tau, m, c):
    Aprime = A * sigma / tau * np.sqrt(np.pi/2)
    mu = center     # For function portability of plot_line and get_line_stats
    y1 = -(x - mu) / tau + sigma**2 / (2 * tau**2)
    y2 = (x - mu) / (np.sqrt(2) * sigma) - sigma / (np.sqrt(2) * tau)
    try:
        yval = Aprime * np.exp(y1) * (1 + erf(y2)) + m*x + c
    except (RuntimeWarning, RuntimeError):
        yval = Aprime * np.exp(y1) * (1 + erf(y2)) + m*x + c
        print(y1, np.exp(y1), y2, erf(y2))

    return yval

def get_flare_type(sigma, tau):
    if sigma / tau < 2.65:
        return 'B'
    else:
        return 'A'
    
def sim_data(xdata, inputparams, noisefree=False):

    yval = exp_gaus_single(xdata, *inputparams) # Base curve
    ydata = np.random.poisson(lam=yval)     # Adding Poisson noise
    pnoise = np.sqrt(ydata)
    wnoise = np.random.normal(0, 2, len(xdata))
    ydata = ydata + wnoise                         # Adding Gaussian noise

    yerr = np.sqrt(pnoise**2 + wnoise**2)   # Calculating total error

    if noisefree:
        return yval, yerr
    else:
        return ydata, yerr