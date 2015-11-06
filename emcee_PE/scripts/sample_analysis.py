#!/usr/bin/env python
# Tools for analysis of MCMC samples from emcee for spin estimation project
#
# MP 04/2015
import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex' : True})

from pydoc import help
from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde
from scipy import stats

import make_runs as mr
from utils import *
import pycbc.pnutils as pnutils


# Constants
quantiles_68 = [0.16, 0.5, 0.84] # 1-sigma
quantiles_95 = [0.0228, 0.5, 0.9772] # 2-sigma ~ 95.4%
percentiles_68 = 100*np.array(quantiles_68)
percentiles_95 = 100*np.array(quantiles_95)
perc_int_95 = [2.28, 97.72]
perc_int_99_7 = [0.13, 99.87] # 3 sigma

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
'xtick.labelsize':11,\
'ytick.labelsize':11,\
'figure.subplot.bottom':0.2,\
'figure.figsize':figsize, \
'savefig.dpi': 300.0, \
'figure.autolayout': True})

# hist(PercentileInterval(s1[0], pc=99.0), 100);

######################################################
# Function to get the bias in the recovered median 
# value for different parameters
######################################################
def get_bias(\
  basedir='/home/prayush/projects/nsbh/TidalParameterEstimation/ParameterBiasVsSnr/SEOBNRv2/set005/TN/',\
  simdir='TN_q2.00_mNS1.35_chiBH0.50_Lambda500.0_SNR60.0/NW100_NS6000/',\
  M_inj = 3*1.35, eta_inj = 2./9., chi1_inj=0, chi2_inj=0.5, SNR_inj = 60):
    """
    Load samples for a given physical configuration, 
    1. decode the location of the corresponding run. 
    2. Compute the 90% confidence interval
    3. Compute the median value of different parameters fromt the posterior.
    """
    #{{{
    test_dir = os.path.join(basedir, simdir)
    m1_inj, m2_inj = pnutils.mchirp_eta_to_mass1_mass2(M_inj * eta_inj**0.6, eta_inj)
    params_inj = {'eta' : eta_inj, 'Mtot' : M_inj, 'Mc' : M_inj * eta_inj**0.6,
                  'chi1' : chi1_inj, 'chi2' : chi2_inj, 
                  'm1' : m1_inj, 'm2' : m2_inj}
    
    match = {}
    match['samples'] = load_samples_join(test_dir, SNR_inj)
    
    bias, CI90, width90 = {}, {}, {}
    for param in ['eta', 'Mc', 'chi1', 'chi2', 'm1', 'm2', 'Mtot']:
      S = match['samples'][param]
      if S == None: return None, [None, None], None
      param_inj = params_inj[param]
      if 'chi' in param: bias[param] = (np.median(S) - param_inj) / 1.
      else: bias[param] = (np.median(S) - param_inj) / param_inj
      CI90[param] = [np.percentile(S, 5), np.percentile(S, 95)]
      if 'chi' in param: width90[param] = CI90[param][1] - CI90[param][0]
      else: width90[param] = (CI90[param][1] - CI90[param][0]) / param_inj    
    #print "bias = %f, width90 = %f" % (bias, width90)
    return bias, CI90, width90
    #}}}

######################################################
# Function to make parameter bias plots
######################################################
linestyles = ['-', '--', '-.', '-x', '--o']
linecolors = ['r', 'g', 'b', 'k', 'm', 'y']

def make_bias_plot(plotparam='Mc', plotquantity='bias', normalize='',\
                    xlabel='$\\rho$', ylabel=None,\
                    savefig='plots/plot.png',\
                    xmin=None, xmax=None, ymin=None, ymax=None):
  #{{{
  gmean = (5.**0.5 + 1.) * 0.5
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
              simdir = mr.get_simdirname(q, mNS, chi2, Lambda, SNR, Nw, Ns)
              print "\n\n Trying to read in %s" % simdir
              chi1val, chi2val = chi1, chi2
              #
              # Hack for NS templates
              if recover_tidal: chi1val, chi2val = chi2, Lambda
              #
              bias, _, width90 = get_bias(simdir=simstring+simdir+'/',\
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
# Set up parameters of signal
######################################################
chi1 = 0.   # small BH
chi2vec = [-0.5, 0, 0.5, 0.74999]  # larger BH
mNS = 1.35
qvec = [2, 3, 4]
Lambdavec = [500, 800, 1000]
SNRvec = [20, 30, 50, 70, 90, 120]


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

inject_tidal = True
recover_tidal= True

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






