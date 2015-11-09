#!/usr/bin/env python
# Tools for analysis of MCMC samples from emcee for spin estimation project
# Copyright 2015
# 
__author__ = "Prayush Kumar <prayush.kumar@ligo.org>"

import os, sys
import commands as cmd
import numpy as np
from matplotlib import mlab, cm, use
#use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex' : True})

from pydoc import help
from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

from utils import *
import pycbc.pnutils as pnutils
from PlotOverlapsNR import make_contour_plot_multrow
import h5py


# Constants
verbose  = True
vverbose = True

quantiles_68 = [0.16, 0.5, 0.84] # 1-sigma
quantiles_95 = [0.0228, 0.5, 0.9772] # 2-sigma ~ 95.4%
percentiles_68 = 100*np.array(quantiles_68)
percentiles_95 = 100*np.array(quantiles_95)
perc_int_95 = [2.28, 97.72]
perc_int_99_7 = [0.13, 99.87] # 3 sigma

CILevels=[90.0, 68.26895, 95.44997, 99.73002]

######################################################
# Function to make parameter bias plots
######################################################
linestyles = ['-', '--', '-.', '-x', '--o']
linecolors = ['r', 'g', 'b', 'k', 'm', 'y']
plotdir = 'plots/'
gmean = (5**0.5 + 1)/2.

# Figure settings
ppi=72.0
aspect=(5.**0.5 - 1) * 0.5
size=4.0 * 2# was 6
figsize=(size,aspect*size)
plt.rcParams.update({\
'legend.fontsize':9, \
'axes.labelsize':11,\
'font.family':'serif',\
'font.size':11,\
'xtick.labelsize':9,\
'ytick.labelsize':9,\
'figure.subplot.bottom':0.2,\
'figure.figsize':figsize, \
'savefig.dpi': 500.0, \
'figure.autolayout': True})

# hist(PercentileInterval(s1[0], pc=99.0), 100);


# Plotting functions
def make_contour_array(X, Y, Z2d, xlabel='Time (s)', ylabel='', clabel='', \
                title='', titles=[], \
                levelspacing=0.25, vmin=None, vmax=None,\
                xmin=None, xmax=None, ymin=None, ymax=None,\
                colorbartype='simple', figname='plot.png'):
  if colorbartype != 'simple':
    raise IOError("Colorbar type %s not supported" % colorbartype)
  colwidth = 2.8*1.5
  if np.shape(X)[:2] != np.shape(Y)[:2] or np.shape(X)[:2] != np.shape(Z2d)[:2]:
    raise IOError("X, Y and Z arrays have different number of sets to plot")
  
  nrow, ncol, _ = np.shape(X)
  if vverbose: print "Making plot with %d rows, %d cols" % (nrow, ncol)
  pltid = 1
  allaxes = []
  fig = plt.figure(int(1e7 * np.random.random()), \
              figsize=(2.*gmean*colwidth*ncol, colwidth*nrow))
  
  for idx in range(nrow):
    for jdx in range(ncol):
      try: xx, yy, zz = X[idx][jdx], Y[idx][jdx], Z2d[idx][jdx]
      except: 
        print "Array shapes = ", np.shape(xx), np.shape(yy), np.shape(zz)
        raise RuntimeError
      if vverbose: print "Adding subplot %d of %f" % (pltid, nrow*ncol)
      ax = fig.add_subplot(nrow, ncol, pltid)
      pltid += 1
      allaxes.append( ax )
      
      norm = cm.colors.Normalize(vmax=zz.max(), vmin=zz.min())
      cmap = cm.rainbow
      levels = np.arange(zz.min(), zz.max(), levelspacing)
      CS = ax.contourf( xx, yy, zz,\
              levels=levels, \
              cmap = cm.get_cmap(cmap, len(levels)-1), norm=norm,\
              alpha=0.9, vmin=vmin, vmax=vmax)
      ax.grid()
      ax.set_xlim([xmin, xmax])
      ax.set_ylim([ymin, ymax])
      if idx == (nrow-1): ax.set_xlabel(xlabel)
      if jdx == 0: ax.set_ylabel(ylabel)
      if np.shape(titles) == (nrow, ncol): ax.set_title(titles[idx][jdx])
      if idx == 0 and jdx==(ncol/2): ax.set_title(title)      
  #
  if colorbartype=='simple':
    ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.7])
    cb = fig.colorbar(CS, cax=ax2, orientation=u'vertical', format='%.1f')
    cb.set_label(clabel)
    cb.set_clim(vmin=vmin, vmax=vmax)
  else:
    # The following code is copied and might not work
    ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.7])
    #ax2 = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    # Make the colorbar
    if type(bounds) == np.ndarray or type(bounds) == list:
      cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
            spacing='uniform', format='%.2f', orientation=u'vertical',\
            ticks=bounds, boundaries=bounds)
    else:
      # How does this colorbar know what colors to span??
      cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
            spacing='uniform', format='%.2f', orientation=u'vertical',\
            ticks=tmp_bounds)
    # Add tick labels
    if logC and (type(bounds) == np.ndarray or type(bounds) == list):
      cb.set_ticklabels(np.round(10**bounds, decimals=4))
    elif type(bounds) == np.ndarray or type(bounds) == list:
      cb.set_ticklabels(np.round(bounds, decimals=4))
    #cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
    #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
    #ax2.set_title(clabel, loc='left')
    cb.set_label(clabel, verticalalignment='top', horizontalalignment='center')#,labelpad=-0.3,y=1.1,x=-0.5)  
    #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])

  fig.savefig(figname)
  return


def make_contour(X, Y, Z2d, xlabel='Time (s)', ylabel='', clabel='', title='',\
                levelspacing=0.25, vmin=None, vmax=None,\
                figname='plot.png'):
  colwidth = 2.8
  plt.figure(int(1e7 * np.random.random()), figsize=(2.*gmean*colwidth, colwidth))
  norm = cm.colors.Normalize(vmax=Z2d.max(), vmin=Z2d.min())
  cmap = cm.rainbow
  levels = np.arange(Z2d.min(), Z2d.max(), levelspacing)
  plt.contourf( X, Y, Z2d,\
              levels=levels, \
              cmap = cm.get_cmap(cmap, len(levels)-1), norm=norm,\
              alpha=0.9, vmin=vmin, vmax=vmax)
  plt.grid()
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  cb = plt.colorbar(format='%.1f')
  cb.set_label(clabel)
  cb.set_clim(vmin=vmin, vmax=vmax)
  plt.savefig(figname)
  return

def make_scatter(X, Y, Z, xlabel='', ylabel='', clabel='', vmin=-1, vmax=None,\
                  figname='plot.png'):
  # Just plot the data
  plt.figure(figsize=(18,6))
  #plt.hexbin(X, Y, C=Z, gridsize=200)
  plt.scatter(X, Y, c=Z, linewidths=0, vmin=vmin, vmax=vmax)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  cb = plt.colorbar()
  cb.set_label(clabel)
  cb.set_clim(vmin=vmin, vmax=vmax)
  plt.grid()
  plt.savefig(figname,dpi=400)

def make_hexbin(X, Y, Z, xlabel='', ylabel='', clabel='', vmin=-1, vmax=None,\
                  figname='plot.png'):
  # Just plot the data
  plt.figure(figsize=(18,6))
  #plt.hexbin(X, Y, C=Z, gridsize=200)
  plt.hexbin(X, Y, C=Z, vmin=vmin, vmax=vmax)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  cb = plt.colorbar()
  cb.set_label(clabel)
  cb.set_clim(vmin=vmin, vmax=vmax)
  plt.grid()
  plt.savefig(figname,dpi=400)


def get_simdirname(q, mNS, chi2, Lambda, SNR, Nw, Ns):
  #{{{
  return 'q%.2f_mNS%.2f_chiBH%.2f_Lambda%.1f_SNR%.1f/NW%d_NS%d'\
              % (q, mNS, chi2, Lambda, SNR, Nw, Ns)
  #}}}

def make_bias_plot(plotparam='Mc', plotquantity='bias', normalize='',\
                    xlabel='$\\rho$', ylabel=None,\
                    savefig='plots/plot.png',\
                    xmin=None, xmax=None, ymin=None, ymax=None):
  #{{{
  colwidth = 2.8
  # parameter bias
  plt.figure(int(1e7 * np.random.random()), figsize=(2.*gmean*colwidth, colwidth))
  for pltid, q in enumerate(qvec):
    plt.subplot(1, len(qvec), pltid + 1)
    for x2idx, chi2 in enumerate(chi2vec):
      for Lidx, Lambda in enumerate(Lambdavec):
        if 'Lambda' in normalize: norm = Lambda
        #elif 'eta' in normalize: norm = q / (1. + q)**2.
        #elif 'Mc' in normalize:
        #  tmp_mt = mNS * (1. + q)
        #  norm = tmp_mt * (q / (1. + q)**2.)**0.6
        else: norm = 1.
        param_bias, param_CIwidth90 = {}, {}
        for SNR in SNRvec:
          for Ns in Nsamples:
            for Nw in Nwalkers:
              simdir = get_simdirname(q, mNS, chi2, Lambda, SNR, Nw, Ns)
              print "\n\n Trying to read in %s" % simdir
              chi1val, chi2val = chi1, chi2
              #
              # Hack for NS templates
              if recover_tidal: chi1val, chi2val = chi2, Lambda
              #
              bias, _, width90 = get_bias(basedir=cmd.getoutput('pwd -P'),\
                      simdir=simstring+simdir+'/',\
                      M_inj=(1.+q)*mNS, eta_inj=q/(1.+q)**2,\
                      chi1_inj=chi1val, chi2_inj=chi2val)
              if bias == None:
                print >> sys.stdout, "Data for %s not found. Ignoring.."\
                                        % (simstring+simdir)
                continue
              for kk in bias:
                if kk in param_bias: param_bias[kk].append(bias[kk])
                else: param_bias[kk] = [bias[kk]]
                if kk in param_CIwidth90: param_CIwidth90[kk].append(width90[kk])
                else: param_CIwidth90[kk] = [width90[kk]]
        # Make the plot legend
        if pltid == 0:
          if Lidx == 0:
            # this happens when x2idx's value changes, ie once for each linestyle
            labelstring = '$\chi_{BH} = %.1f$' % (chi2)
          else: labelstring=''
        elif pltid == 1:
          if x2idx == 0:
            # this happens when Lidx = 0,1,2, i.e. once for each linecolor
            labelstring = '$\Lambda = %.0f$' % (Lambda)
          else: labelstring = ''
        # Make the acual plot
        if plotquantity == 'bias': 
          plotvec = np.array(param_bias[plotparam])/norm
        elif plotquantity == 'width90' or 'width' in plotquantity: 
          plotvec = np.array(param_CIwidth90[plotparam])/norm
        if labelstring == '':
          plt.plot(SNRvec, plotvec, linecolors[Lidx]+linestyles[x2idx],lw=2)
        else:
          plt.plot(SNRvec, plotvec,\
                  linecolors[Lidx]+linestyles[x2idx],\
                  label=labelstring,lw=2)      
    #
    plt.legend(loc='best')
    plt.grid()
    if xlabel: plt.xlabel(xlabel)
    if pltid == 0 and ylabel: plt.ylabel(ylabel)
    if xmin is not None: plt.xlim(xmin=xmin)
    if xmax is not None: plt.xlim(xmax=xmax)
    if ymin is not None: plt.ylim(ymin=ymin)
    if ymax is not None: plt.ylim(ymax=ymax)
    plt.title('$q = %.1f$' % q)
  plt.savefig(savefig)
  #}}}


######################################################
# Function to get the bias in the recovered median 
# value for different parameters
######################################################


# Function to get the exact parameter bias/whatever given the injection
# parameters as input
def get_results(data, q=None, chiBH=None, NSLmbd=None, SNR=None,\
        p='Mc', qnt='median-val', \
        CI=0, CILevs=[90.0, 68.26895, 95.44997, 99.73002]):
  #{{{
  '''
  Confidence interval is indexed as {0 : 90, 1 : 68%, 2 : 95%, 3 : 99.7%}
  Parameter requested in **p** has to be one of :
  * 'm1'
  * 'm2'
  * 'Mc'
  * 'Mtot'
  * 'eta'
  * 'chiBH'
  * 'Lambda'

  Quantity requested in **qnt** has to be one of :
  * 'median-val': X(median)
  * 'fbias'     : (X(median) - X(injection)) / X(injection)
  * 'CIlower'   : X[10%] or 100-x%
  * 'CIhigher'  : X[90%] or x%
  * 'CIfwidth'  : (X[90%] - X[10%]) / X(injection)
  '''
  if q is None or chiBH is None or NSLmbd is None or SNR is None:
    raise IOError("Need all of (q,chi,NSL,SNR) parameters")
  l1grp, l2grp, l3grp = \
        ['q%.1f.dir' % q, 'chiBH%.2f.dir' % chiBH, 'LambdaNS%.1f.dir' % NSLmbd]
  dset = 'SNR%.1f.dat' % SNR
  if vverbose: print "Group names = ", l1grp, l2grp, l3grp, dset
  alldata = data[l1grp][l2grp][l3grp][dset].value
  
  num_of_data_fields = 5
  if 'm1' in p: pidx = 2 + 0*num_of_data_fields
  elif 'm2' in p: pidx = 2 + 1*num_of_data_fields
  elif 'Mc' in p: pidx = 2 + 2*num_of_data_fields
  elif 'Mtot' in p: pidx = 2 + 3*num_of_data_fields
  elif 'eta' in p: pidx = 2 + 4*num_of_data_fields
  elif 'chiBH' in p: pidx = 2 + 5*num_of_data_fields
  elif 'Lambda' in p: pidx = 2 + 6*num_of_data_fields
  
  if 'median-val' in qnt: pidx += 0
  elif 'fbias' in qnt: pidx += 1
  elif 'CIlower' in qnt: pidx += 2
  elif 'CIhigher' in qnt: pidx += 3
  elif 'CIfwidth' in qnt: pidx += 4
  
  return alldata[CI, pidx]
  #}}}

# Function to get statistical properties of recovered parameters as functions 
# of the injected SNR
def get_results_vs_snr(data, q=None, chiBH=None, NSLmbd=None,\
              p='Mc', qnt='CIfwidth', CI=0,\
              SNRvec=[20, 30, 50, 70, 90, 120]):
  #{{{
  """
  This function returns the requested quantity for the requested parameter
  as a function of the SNR from the list of input SNRs.
  
  Confidence interval is indexed as {0 : 90, 1 : 68%, 2 : 95%, 3 : 99.7%}
  Parameter requested in **p** has to be one of :
  * 'm1'
  * 'm2'
  * 'Mc'
  * 'Mtot'
  * 'eta'
  * 'chiBH'
  * 'Lambda'

  Quantity requested in **qnt** has to be one of :
  * 'median-val': X(median)
  * 'fbias'     : (X(median) - X(injection)) / X(injection)
  * 'CIlower'   : X[10%] or 100-x%
  * 'CIhigher'  : X[90%] or x%
  * 'CIfwidth'  : (X[90%] - X[10%]) / X(injection)
  """
  param_values = np.array([])
  for snr in SNRvec:
    param_values = np.append( param_values, \
        get_results(data, q=q, chiBH=chiBH, NSLmbd=NSLmbd, SNR=snr,\
                    p=p, qnt=qnt, CI=CI) )
  return np.array(SNRvec), np.array(param_values)
  #}}}

# Function to find out when a given statistical property of a recovered 
# parameter attains a target value
def get_snr_where_quantity_val_reached(data, q=None, chiBH=None, NSLmbd=None,\
              p='Lambda', qnt='CIfwidth', CI=0, target_val=1., \
              SNRvec=[20, 30, 50, 70, 90, 120]):
  #{{{
  """
  This function :
  - Calculates the requested quantitiy as a function of injected SNR
  - Calculates the SNR threshold where this quantity attains the required value
  - Returns this SNR
  
  Confidence interval is indexed as {0 : 90, 1 : 68%, 2 : 95%, 3 : 99.7%}
  Parameter requested in **p** has to be one of :
  * 'm1'
  * 'm2'
  * 'Mc'
  * 'Mtot'
  * 'eta'
  * 'chiBH'
  * 'Lambda'

  Quantity requested in **qnt** has to be one of :
  * 'median-val': X(median)
  * 'fbias'     : (X(median) - X(injection)) / X(injection)
  * 'CIlower'   : X[10%] or 100-x%
  * 'CIhigher'  : X[90%] or x%
  * 'CIfwidth'  : (X[90%] - X[10%]) / X(injection)
  
  Desired level of the value is given in **target_value**
  """
  snr, values = get_results_vs_snr(data, q=q, chiBH=chiBH, NSLmbd=NSLmbd,\
                    p=p, qnt=qnt, CI=CI, SNRvec=SNRvec)
  valuesI = UnivariateSpline(snr, values)
  
  def fun(snr): return np.abs(valuesI(snr) - target_val)
  result = minimize_scalar(fun, bounds=[snr.min(), snr.max()], method='Bounded')
  
  return result['x'], result['fun']
  #}}}


######################################################
######################################################
# BEGIN
######################################################
######################################################

######################################################
# INPUTS
######################################################
# Injection / Recovery information
if len(sys.argv) >= 3:
  if int(sys.argv[1]) != 0: inject_tidal = True
  else: inject_tidal = False
  #
  if int(sys.argv[2]) != 0: recover_tidal = True
  else: recover_tidal = False

# plotting flags
if len(sys.argv) >= 5:
  if int(sys.argv[3]) != 0: plot_bias = True
  else: plot_bias = False
  #
  if int(sys.argv[4]) != 0: plot_width = True
  else: plot_width = False

######################################################
# Set up parameters of signal
######################################################
chi1 = 0.   # small BH
chi2vec = [-0.5, 0, 0.5, 0.74999]  # larger BH
mNS = 1.35
qvec = [2, 3, 4]
Lambdavec = [0]#[500, 800, 1000]
SNRvec = [20, 30, 50, 70, 90, 120]
if inject_tidal: Lambdavec = [500, 800, 1000]

######################################################
# Set up parameters of templates
######################################################
# Taking an uninformed prior
m1min = 1.2
m1max = 15.
m2min = 1.2 
m2max = 25

LambdaMax = 4000
Lambdastdev = 100

######################################################
# Set up RUN parameters
######################################################
if inject_tidal: sigstring = 'T'
else: sigstring = 'N'
if recover_tidal: tmpstring = 'T'
else: tmpstring = 'N'
simstring = sigstring + tmpstring + '_'

Nwalkers = [100]
Nsamples = [150000]
Nburnin  = 500

######################################################
# Read in parameter biases and other data
######################################################
datafile = 'TT_ParameterBiasesAndConfidenceIntervals.h5'
data = h5py.File(datafile, 'r')

print \
'''
The data stored in the HDF file has the following format:
- Level1 groups : Mass-ratio q. name-format is ('q%.1f.dir' % q)
- Level2 groups : Black-hole spins. name-format is ('chiBH%.2f.dir' % chiBH)
- Level3 groups : Neutron star Lambda. name-format is ('LambdaNS%.1f.dir' % LambdaNS)
- Dataset : Injection's SNR. name-format is ('SNR%.1f.dat' % SNR)
'''
print \
'''
The dataset is a 2-d array, with rows corresponding to confidence-levels. 
We use [90%, 1sigma, 2sigma, 3sigma] confidence probabilities, and have the 
confidence interval associated with the posterior of each of the binary's 
physical parameters, i.e. ['m1', 'm2', 'Mc', 'Mtot', 'eta', 'chi2', 'Lambda']

We have stored 5 quantities for each parameter k:
k+0: Median recovered parameter value from posterior samples
k+1: Bias = (X(median) - X(inj))/X(inj)
k+2: Confidence level lower bound = X(confidence_level_low)
k+3: Confidence level upper bound = X(confidence_level_up)
k+4: Confidence interval = (X(confidence_level_up) - X(confidence_level_low))/X(inj)
    
PLUS, the first 2 columns in each row of the dataset are:
0 : Confidence interval probability (actual number \in [0,1])
1 : max LogLikelihood value (actually the match value, which can be 
    converted to LogLikelihood as LogL = 1/2 SNR^2 MATCH^2
'''

plot_SNRcrit = True
plot_LambdaBias = True

######################################################
######################################################
# Time to MAKE PLOTS
######################################################
######################################################
print "MAKING PLOTS NOW.."

print \
"""
Plotting the SNR threshold below which our measurement error on the NS Lambda
parameter are 100%, i.e. the SRN below which we cannot make statements about
the tidal deformability of the Neutron Star
"""

error_threshold = 1
snrThresholds = {}

for Lambda in Lambdavec:
  snrThresholds[Lambda] = {}
  for CI in range( len(CILevels) ):
    snrThresholds[Lambda][CI] = np.zeros( (len(qvec), len(chi2vec)) )
    for i, q in enumerate(qvec):
      for j, chiBH in enumerate(chi2vec):
        if vverbose:
          print "getting SNR thresholds for q=%f, chiBh=%f, Lambda=%f" % \
                      (q, chiBH, Lambda)
        snrThresholds[Lambda][CI][i,j], fn =get_snr_where_quantity_val_reached(\
            data, q=q, chiBH=chiBH, NSLmbd=Lambda, \
            p='Lambda', qnt='CIfwidth', CI=CI, target_val=error_threshold,\
            SNRvec=SNRvec)
        if fn > 1.e-2:
          print "\n\n warning : Lambda only bound to %f %%" % (100*(fn+1))
          print "\t for q=%f, chiBh=%f, Lambda=%f" % (q, chiBH, Lambda)
          snrThresholds[Lambda][CI][i,j] = np.max(SNRvec)

if plot_SNRcrit:
  for Lambda in Lambdavec:
    for CI in range( len(CILevels) ):
      if verbose:
        print "Plotting for Lambda injected = %f, at Confidence level = %f" % \
                (Lambda, CILevels[CI])
      
      snrThresh = snrThresholds[Lambda][CI]
      make_contour(chi2vec, qvec, snrThresh,\
        xlabel='Black-hole spin', ylabel='Binary mass-ratio',\
        clabel='SNR below which $\delta\Lambda_\mathrm{NS}\sim 100\%$',\
        title='$\Lambda_\mathrm{NS}^\mathrm{Injected}=%.1f$, at %.1f%% confidence' %\
          (Lambda, CILevels[CI]),\
        levelspacing=2, \
        vmin=snrThresh.min(), vmax=snrThresh.max(), \
        figname=os.path.join(plotdir, simstring+\
      'SNRThresholdForLambdaMeasurement_BHspin_MassRatio_Lambda%.1f_CI%.1f.png' %\
          (Lambda, CILevels[CI])))
      #
      make_contour(chi2vec, np.array(qvec) * mNS, snrThresh,\
        xlabel='Black-hole spin', ylabel='Black-hole mass $(M_\odot)$',\
        clabel='SNR below which $\delta\Lambda_\mathrm{NS}\sim 100\%$',\
        title='$\Lambda_\mathrm{NS}^\mathrm{Injected}=%.1f$, at %.1f%% confidence' %\
          (Lambda, CILevels[CI]),\
        levelspacing=2, \
        vmin=snrThresh.min(), vmax=snrThresh.max(), \
        figname=os.path.join(plotdir, simstring+\
      'SNRThresholdForLambdaMeasurement_BHspin_BHmass_Lambda%.1f_CI%.1f.png' %\
          (Lambda, CILevels[CI])))



print \
"""
Plotting the fractional bias in Lambda recovery at different SNR levels, as a 
function of the BH mass and spin. 

The bias shown is a fraction of the injected Lambda, and different figures are
made for different LAmbda values
"""

lambdaBiases = {}
for Lambda in Lambdavec:
  lambdaBiases[Lambda] = {}
  for snr in SNRvec:
    lambdaBiases[Lambda][snr] = {}
    for CI in range( len(CILevels) ):
      lambdaBiases[Lambda][snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
      for i, q in enumerate(qvec):
        for j, chiBH in enumerate(chi2vec):
          if verbose:
            print "getting bias in Lambda for q=%f, chiBh=%f, Lambda=%f at SNR = %f" %\
                  (q, chiBH, Lambda, snr)
          lambdaBiases[Lambda][snr][CI][i,j] = get_results(data,\
                  q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                  p='Lambda', qnt='fbias', CI=CI)

if plot_LambdaBias:
  plotSNRvec = [30, 50, 90]
  plotCI = 0
  Xarray, Yarray, Zarray = [], [], []
  titles = []
  
  for Lambda in Lambdavec:
    Xtmp, Ytmp, Ztmp = [], [], []
    ttmp = []
    for snr in plotSNRvec:
      Xtmp.append(np.array(chi2vec))
      Ytmp.append(np.array(qvec))
      Ztmp.append(lambdaBiases[Lambda][snr][plotCI] * 100)
      ttmp.append('$\Lambda_\mathrm{NS}=%.1f, \rho=%.1f$' % (Lambda, snr))
    Xarray.append(Xtmp)
    Yarray.append(Ytmp)
    Zarray.append(Ztmp)
    titles.append(ttmp)
  
  make_contour_array(Xarray, Yarray, Zarray, \
    xlabel='Black-hole spin', ylabel='Binary mass-ratio', \
    xmin=-0.5, xmax=0.75, ymin=2, ymax=4, titles=titles, \
    clabel='$\%$ bias in $\Lambda_\mathrm{NS}$', levelspacing=0.02, \
    figname=os.path.join(plotdir,'TT_LambdaBiases_Lambda_SNR.png'))
















































































######################################################
# Make parameter bias plots
######################################################
#############################
##### Chirp mass bias
if plot_bias:
  print "Making chirp mass bias plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = -0.0002, 0.00035
  make_bias_plot(plotparam='Mc', ylabel='$\Delta\mathcal{M}_c / \mathcal{M}_c $',\
              savefig='plots/'+simstring+'chirpMassBias_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# Chirp mass width
if plot_width:
  print "Making chirp mass CI width plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = 0.0, 0.0012
  make_bias_plot(plotparam='Mc', plotquantity='width',\
              ylabel='$\left(\Delta\mathcal{M}_c\\right)_{90\%} / \mathcal{M}_c$',\
              savefig='plots/'+simstring+'chirpMassCIWidth90_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

#############################
##### eta bias
if plot_bias:
  print "Making eta bias plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = -0.1, 0.06
  make_bias_plot(plotparam='eta', ylabel='$\Delta\eta / \eta $',\
              savefig='plots/'+simstring+'EtaBias_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# eta width
if plot_width:
  print "Making eta CI width plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = 0.02, 0.36
  make_bias_plot(plotparam='eta', plotquantity='width',\
              ylabel='$\left(\Delta\eta\\right)_{90\%} / \eta $',\
              savefig='plots/'+simstring+'EtaCIWidth90_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

#############################
##### Chi-BH bias
if recover_tidal: plottag = 'chi1'
else: plottag = 'chi2'
if plot_bias:
  print "Making BH spin bias plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = -0.3, 0.15
  make_bias_plot(plotparam=plottag, ylabel='$\Delta \chi_\mathrm{BH}$',\
              savefig='plots/'+simstring+'BHspinBias_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# Chi-BH width
if plot_width:
  print "Making BH spin CI width plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = 0.0, 0.9
  make_bias_plot(plotparam=plottag, plotquantity='width',\
              ylabel='$\left(\Delta \chi_\mathrm{BH}\\right)_{90\%}$',\
              savefig='plots/'+simstring+'BHspinCIWidth90_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


#############################
# Lambda-NS bias & width
if recover_tidal:
  plottag = 'chi2'
  if plot_bias:
    print "Making NS Lambda bias plots"
    xmin, xmax = min(SNRvec), max(SNRvec)
    ymin, ymax = -0.2, 2.5
    make_bias_plot(plotparam=plottag,\
              normalize='Lambda',\
              ylabel='$\Delta\Lambda_\mathrm{NS} / \Lambda_\mathrm{NS}$',\
              savefig='plots/'+simstring+'NSLambdaBias_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
  
  if plot_width:
    print "Making NS Lambda CI Width plots"
    xmin, xmax = min(SNRvec), max(SNRvec)
    ymin, ymax = 0, 7
    make_bias_plot(plotparam=plottag, plotquantity='width',\
        normalize='Lambda',\
        ylabel='$\left(\Delta\Lambda_\mathrm{NS}\\right)_{90\%} / \Lambda_\mathrm{NS}$',\
              savefig='plots/'+simstring+'NSLambdaCIWidth90_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)






