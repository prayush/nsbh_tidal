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
plt.rcParams.update({'legend.fontsize':9, 'axes.labelsize':11,'font.family':'serif','font.size':11,'xtick.labelsize':11,'ytick.labelsize':11,'figure.subplot.bottom':0.2,'figure.figsize':figsize, 'savefig.dpi': 150.0, 'figure.autolayout': True})


# Parameter transformation functions
def etafun(q):
  return q/(1.0 + q)**2
def qfun(eta):
  return (1.0 + np.sqrt(1.0 - 4.0*eta) - 2.0*eta) / (2.0*eta)
def m1fun(M,q):
  return M*1.0/(1.0+q)
def m2fun(M,q):
  return M*q/(1.0+q)
def Mchirpfun(M, eta):
  return M*eta**(3.0/5.0)
def Mfun(Mc, eta):
  return Mc*eta**(-3.0/5.0)

def chi_eff(eta, chi1, chi2):
    #{{{
    q = qfun(eta)
    M = 1
    m1 = m1fun(M,q)
    m2 = m2fun(M,q) # m2 >= m1 as in python code
    chi_s = (chi1 + chi2)/2.0;
    chi_a = (chi1 - chi2)/2.0;
    delta = (m1-m2)/M;
    return chi_s + delta * chi_a
    #}}}

def chi_PN(eta, chi1, chi2):
    #{{{
    q = qfun(eta)
    M = 1
    m1 = m1fun(M,q)
    m2 = m2fun(M,q) # m2 >= m1 as in python code
    chi_s = (chi1 + chi2)/2.0;
    chi_a = (chi1 - chi2)/2.0;
    delta = (m1-m2)/M;
    return chi_s * (1.0 - 76.0/113.0 * eta) + delta * chi_a # Ajith's PN reduced spin parameter
    #}}}

# Function to remove duplicate entries from a list
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x))]

def combine_pickles(all_data_files):
    #{{{
    all_pickles = [np.load(fnam) for fnam in all_data_files]
    pid = 0
    for pickle in all_pickles:
      if np.shape(np.shape(pickle))[0] != 3:
        if verbose: 
          print >>sys.stdout, "Skipping subfile %s" % all_data_files[pid]
        continue
      if pid == 0:
        combined = pickle
      else: combined = np.append( combined, pickle, axis=0 )
    #
    return combined
    #}}}

def read_run_part_names(dataDir, SNR, burnin=500, useMaxNRun=True, \
                      chain_number=-1, verbose=True):
    #{{{
    import glob
    # Figure out the random integer associated with each run
    all_files = glob.glob(dataDir + "/chain*-*.npy")
    chain_nums= [int(f.split('/')[-1].split('-')[0].strip('chain')) for f in all_files]
    chain_nums = f7(chain_nums)
    # Get all continuation files for each
    part_files = {}
    for chnum in chain_nums:
      tmp = glob.glob(dataDir + ("/chain%d-?.npy" % chnum))
      tmp.sort()
      part_files[chnum] = tmp
      tmp = glob.glob(dataDir + ("/chain%d-??.npy" % chnum))
      tmp.sort()
      part_files[chnum].extend( tmp )
      tmp = glob.glob(dataDir + ("/chain%d-???.npy" % chnum))
      tmp.sort()
      part_files[chnum].extend( tmp )
    # Which run to use?
    if useMaxNRun:
      last_len, last_chnum = -1, -1
      for chnum in part_files:
        if len(part_files[chnum]) > last_len:
          last_len = len(part_files[chnum])
          last_chnum = chnum
    else: 
      last_chnum, last_len = chain_number, len(part_files[chain_number])
    if verbose: 
      print >>sys.stdout, "run %d, %d subfiles" % (last_chnum, last_len)
    # If no runs are found
    if last_len <= 0 or last_chnum <= 0:
      return [], []
    # Now open each continuation file for the chosen run and gather samples
    #print part_files[last_chnum]
    part_files_loglike = [fnam.replace('chain','loglike') for fnam in part_files[last_chnum]]    
    #chain = combine_pickles( part_files[last_chnum] )
    #loglike = combine_pickles( part_files_loglike )
    chain = np.load(part_files[last_chnum][last_len-1])
    loglike = np.load(part_files_loglike[last_len-1])
    #
    if verbose:
      print >>sys.stdout, \
        "reading from %s/%s" % (part_files[last_chnum][last_len-1].split('/')[-3],\
                                part_files[last_chnum][last_len-1].split('/')[-1])
      print >>sys.stdout, \
        "samples (chain) = ", np.shape(chain), " (loglike) = ", np.shape(loglike)
    return chain, loglike
    #}}}

def load_samples_join(dataDir, SNR, burnin=500, useMaxNRun=True, \
                      chain_number=-1, verbose=True):
    """
    Load samples from double-spin emcee run. 
    Throw away samples before burnin and at very low likelihood. Convert to match.
    Expect several runs with N continuation files, i.e. chainXXX-1.npy, 
    chainXXX-2.npy, ... chainXXX-N.npy. Each file has 1000 steps
    Choose whichever run has the most continuation files
    Option to combine samples from different runs as well
    """
    #{{{
    import glob
    chain, loglike = read_run_part_names(dataDir, SNR, burnin=burnin, \
                                        useMaxNRun=useMaxNRun, \
                                        chain_number=chain_number, \
                                        verbose=verbose )
    if np.shape(chain)[0] != np.shape(loglike)[1] or np.shape(chain)[1] != np.shape(loglike)[0]:
      raise RuntimeError("Chain and loglikelihood file not the same length")
    if len(chain) == 0 or len(loglike) == 0:
      if verbose:
        print "Empty lists gotten for chain and loglike for %s" % dataDir
      return {'eta' : None, 'Mc' : None, 'chieff' : None, 'chiPN' : None,
              'chi1' : None, 'chi2' : None, 'Mtot' : None,
              'm1' : None, 'm2' : None, 'match' : None}
      #raise RuntimeError("Empty lists gotten for chain and loglike")
    #
    # Here onwards code remains the same as for load_samples
    
    samples_ok = chain[:, burnin:, :]
    loglike_ok = loglike[burnin:,:] # it's really log posterior pdf, but can get get back match modulo prior
    sel = np.isfinite(loglike_ok)  # could put something more stringent here!
    etaval = samples_ok[:,:,0].T[sel]
    Mcval = samples_ok[:,:,1].T[sel]
    chi1val = samples_ok[:,:,2].T[sel]
    chi2val = samples_ok[:,:,3].T[sel]
    loglikeval = loglike_ok[sel]
    
    mask = loglikeval > 0.2*max(loglikeval) # throw away samples at very low loglikelihood
    loglikeval = loglikeval[mask]
    etaval = etaval[mask]
    Mcval = Mcval[mask]
    chi1val = chi1val[mask]
    chi2val = chi2val[mask]
    
    matchval = np.sqrt(2*loglikeval)/SNR # convert to match
    
    qval = qfun(etaval)
    Mval = Mfun(Mcval,etaval)
    m1val = m1fun(Mval,qval)
    m2val = m2fun(Mval,qval)
    chieffval = chi_eff(etaval, chi1val, chi2val)
    chiPNval = chi_PN(etaval, chi1val, chi2val)
    
    return {'eta' : etaval, 'Mc' : Mcval, 'chieff' : chieffval, 'chiPN' : chiPNval, 
            'chi1' : chi1val, 'chi2' : chi2val, 'Mtot' : Mval, 
            'm1' : m1val, 'm2' : m2val, 'match' : matchval}
    #}}}

def load_samples(dataDir, SNR, burnin=500):
    """
    Load samples from double-spin emcee run. Throw away samples before burnin and at very low likelihood. Convert to match.
    """
    #{{{
    chfile = dataDir+'/chain.npy'
    llfile = dataDir+'/loglike.npy'
    if not os.path.exists(chfile) or not os.path.exists(llfile):
      return {'eta' : None, 'Mc' : None, 'chieff' : None, 'chiPN' : None,
              'chi1' : None, 'chi2' : None, 'Mtot' : None,
              'm1' : None, 'm2' : None, 'match' : None}

    chain = np.load(chfile)
    loglike = np.load(llfile)
    
    samples_ok = chain[:, burnin:, :]
    loglike_ok = loglike[burnin:,:] # it's really log posterior pdf, but can get get back match modulo prior
    sel = np.isfinite(loglike_ok)  # could put something more stringent here!
    etaval = samples_ok[:,:,0].T[sel]
    Mcval = samples_ok[:,:,1].T[sel]
    chi1val = samples_ok[:,:,2].T[sel]
    chi2val = samples_ok[:,:,3].T[sel]
    loglikeval = loglike_ok[sel]
    
    mask = loglikeval > 0.2*max(loglikeval) # throw away samples at very low loglikelihood
    loglikeval = loglikeval[mask]
    etaval = etaval[mask]
    Mcval = Mcval[mask]
    chi1val = chi1val[mask]
    chi2val = chi2val[mask]
    
    matchval = np.sqrt(2*loglikeval)/SNR # convert to match
    
    qval = qfun(etaval)
    Mval = Mfun(Mcval,etaval)
    m1val = m1fun(Mval,qval)
    m2val = m2fun(Mval,qval)
    chieffval = chi_eff(etaval, chi1val, chi2val)
    chiPNval = chi_PN(etaval, chi1val, chi2val)
    
    return {'eta' : etaval, 'Mc' : Mcval, 'chieff' : chieffval, 'chiPN' : chiPNval, 
            'chi1' : chi1val, 'chi2' : chi2val, 'Mtot' : Mval, 
            'm1' : m1val, 'm2' : m2val, 'match' : matchval}
    #}}}

def PercentileInterval(s, pc=95.0):
    #{{{
    offset = (100.0-pc)/2.0
    lowbound = s > np.percentile(s, offset)
    hibound = s < np.percentile(s, 100.0 - offset)
    mask = np.logical_and(lowbound, hibound)
    return s[mask]
    #}}}

# hist(PercentileInterval(s1[0], pc=99.0), 100);

# simple example

def get_bias(basedir = '/home/prayush/projects/nsbh/TidalParameterEstimation/ParameterBiasVsSnr/SEOBNRv2/set003/',\
  simdir='TN_q2.00_mNS1.35_chiBH0.50_Lambda500.0_SNR60.0/NW100_NS6000/',\
  M_inj = 3*1.35, eta_inj = 2./9., chi1_inj=0, chi2_inj=0.5, SNR_inj = 60):
    """
    Load samples for a given physical configuration, by first decoding the location
    of the corresponding run. Compute the 90% confidence interval and the median
    value of different parameters fromt the posterior. Return both.
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
chi2vec = [-0.5, 0, 0.5]  # larger BH
mNS = 1.35
qvec = [2, 3]
Lambdavec = [500, 1000, 2000]
#SNRvec = [20, 30, 40, 50, 60, 70, 80, 90, 100]
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
#if plot_bias:
  #print "Making chirp mass bias plots"
  #xmin, xmax = min(SNRvec), max(SNRvec)
  #ymin, ymax = -0.0002, 0.00035
  #make_bias_plot(plotparam='Mc', ylabel='$\Delta\mathcal{M}_c / \mathcal{M}_c $',\
              #savefig='plots/'+simstring+'chirpMassBias_vs_SNR_q23.pdf',\
              #xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

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
#if plot_bias:
  #print "Making eta bias plots"
  #xmin, xmax = min(SNRvec), max(SNRvec)
  #ymin, ymax = -0.1, 0.06
  #make_bias_plot(plotparam='eta', ylabel='$\Delta\eta / \eta $',\
              #savefig='plots/'+simstring+'EtaBias_vs_SNR_q23.pdf',\
              #xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

## eta width
#if plot_width:
  #print "Making eta CI width plots"
  #xmin, xmax = min(SNRvec), max(SNRvec)
  #ymin, ymax = 0.02, 0.36
  #make_bias_plot(plotparam='eta', plotquantity='width',\
              #ylabel='$\left(\Delta\eta\\right)_{90\%} / \eta $',\
              #savefig='plots/'+simstring+'EtaCIWidth90_vs_SNR_q23.pdf',\
              #xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

#############################
##### Chi-BH bias
if recover_tidal: plottag = 'chi1'
else: plottag = 'chi2'
#if plot_bias:
  #print "Making BH spin bias plots"
  #xmin, xmax = min(SNRvec), max(SNRvec)
  #ymin, ymax = -0.3, 0.15
  #make_bias_plot(plotparam=plottag, ylabel='$\Delta \chi_\mathrm{BH}$',\
              #savefig='plots/'+simstring+'BHspinBias_vs_SNR_q23.pdf',\
              #xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

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
#if recover_tidal:
  #plottag = 'chi2'
  #if plot_bias:
    #print "Making NS Lambda bias plots"
    #xmin, xmax = min(SNRvec), max(SNRvec)
    #ymin, ymax = -0.2, 2.5
    #make_bias_plot(plotparam=plottag,\
              #normalize='Lambda',\
              #ylabel='$\Delta\Lambda_\mathrm{NS} / \Lambda_\mathrm{NS}$',\
              #savefig='plots/'+simstring+'NSLambdaBias_vs_SNR_q23.pdf',\
              #xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
  
  #if plot_width:
    #print "Making NS Lambda CI Width plots"
    #xmin, xmax = min(SNRvec), max(SNRvec)
    #ymin, ymax = 0, 7
    #make_bias_plot(plotparam=plottag, plotquantity='width',\
        #normalize='Lambda',\
        #ylabel='$\left(\Delta\Lambda_\mathrm{NS}\\right)_{90\%} / \Lambda_\mathrm{NS}$',\
              #savefig='plots/'+simstring+'NSLambdaCIWidth90_vs_SNR_q23.pdf',\
              #xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)






