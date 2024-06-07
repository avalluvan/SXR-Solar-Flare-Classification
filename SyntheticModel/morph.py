import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cycler
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import curve_fit
from scipy.special import erf, seterr, SpecialFunctionError, SpecialFunctionWarning

defaultcolor = '#002060'
plt.rcParams.update({'text.color': defaultcolor, 'axes.labelcolor': defaultcolor, 
                     'xtick.color': defaultcolor, 'ytick.color': defaultcolor,
                     'axes.prop_cycle': cycler(color=['b', 'r', 'limegreen']),
                     'font.family':'serif', 'font.serif': 'Times New Roman',
                     'font.size': 22, 'lines.linewidth': 3,
                     'figure.figsize': (9.6, 5.4), 'figure.dpi': 100})

# np.seterr(divide='print', over='raise', under='raise', invalid='raise')
np.seterr(all='raise')
# seterr(all='warn')

def exp_gaus_single(x, A, center, sigma, tau, m, c):
    Aprime = A * sigma / tau * np.sqrt(np.pi/2)
    mu = center     # For function portability of plot_line and get_line_stats
    y1 = -(x - mu) / tau + sigma**2 / (2 * tau**2)
    y2 = (x - mu) / (np.sqrt(2) * sigma) - sigma / (np.sqrt(2) * tau)
    try:
        yval = Aprime * np.exp(y1) * (1 + erf(y2)) + m*x + c
    except (RuntimeWarning, RuntimeError, FloatingPointError, SpecialFunctionError) as f:
        print(f)
        return 'error'

    return yval

def get_flare_type(sigma, tau):
    if sigma / tau < 2.65:
        return 'B'
    else:
        return 'A'
    
def sim_data(xdata, inputparams, noisefree=False):

    yval = exp_gaus_single(xdata, *inputparams) # Base curve
    if type(yval) == str:
        print('yval returned error.')
        return 'error', 'error'
    ydata = np.random.poisson(lam=yval)     # Adding Poisson noise
    pnoise = np.sqrt(ydata)
    wnoise = np.random.normal(0, 2, len(xdata))
    ydata = ydata + wnoise                         # Adding Gaussian noise

    yerr = np.sqrt(pnoise**2 + wnoise**2)   # Calculating total error

    if noisefree:
        return yval+1e-3, yerr
    else:
        return ydata, yerr
    
def _plot_flare(xt, yt, yerr, popt, pcov, inputparams):
    fig, axs = plt.subplots(2, figsize=(9.6, 9.6), gridspec_kw={'height_ratios': [3, 1]})
        
    plt.xlabel('Time (s)')
    plt.subplots_adjust(hspace=.0)
    linestyle = {"elinewidth":1, "capsize":0, "ecolor":"grey"}
    plt.setp(axs[0].get_xticklabels(), visible=False)

    axs[0].errorbar(xt, yt, yerr=yerr, fmt='D', ms=3, c='blue', **linestyle)

    for ax in axs:
        ax.tick_params(which='major', width=1)
        ax.tick_params(which='major', length=4)
        ax.margins(x=0)
    axs[0].set_ylabel('Flux (nW/m$^2$)')
    axs[1].set_ylabel('Residual ((o-f)/$\\sigma$)')
    axs[1].set_ylim(bottom=-4,top=4)

    yfit = exp_gaus_single(xt, *popt)
    axs[0].plot(xt, yfit, c='r', zorder=10, lw=3)
    axs[0].tick_params(axis='x', direction='in')
    axs[1].errorbar(xt, (yt-yfit)/np.std(yt-yfit), yerr=yerr/np.std(yt-yfit), fmt='D', ms=3, c='blue', **linestyle)
    axs[1].axhline(np.mean((yt-yfit)/np.std(yt-yfit)), linestyle='--', lw=1, c='navy')
    
    at = AnchoredText(f'Input σ/τ = {inputparams[2]/inputparams[3]:.2f}\nFit σ/τ = {popt[2]/popt[3]:.2f}', \
            prop=dict(size=18, color='black'), frameon=False, loc='upper left')
    at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    axs[0].add_artist(at)

    at = AnchoredText(f'A = {inputparams[0]:.2f}| {popt[0]:.2f}\n\
                        $\\mu$ = {inputparams[1]:.2f}| {popt[1]:.2f}\n\
                        σ = {inputparams[2]:.2f}| {popt[2]:.2f}\n\
                        τ = {inputparams[3]:.2f}| {popt[3]:.2f}\n\
                        m = {inputparams[4]:.2f}| {popt[4]:.2f}\n\
                        c = {inputparams[5]:.2f}| {popt[5]:.2f}\n', \
            prop=dict(size=18, color='black'), frameon=False, loc='upper right')
    at.patch.set_boxstyle('round,pad=0.,rounding_size=0.2')
    axs[0].add_artist(at)

    return popt, plt.gcf()
    
def plot_flare(time, counts, error, inputparams, plot=False):

    popt, pcov = curve_fit(exp_gaus_single, time, counts, p0=inputparams)

    if plot == True:
        popt, fig = _plot_flare(time, counts, error, popt, pcov, inputparams)
        return popt, fig
    else:
        return popt, pcov