#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" 
this file contains miscellanous utility functions used by the 
MCMC python code using emcee to compute 4D credible regions
This version uses parameters [eta, Mc, chi1, chi2] for SEOBNRv2 ROM templates.

The idea is to use a match-based (zero-noise) likelihood 
L(s, theta) = exp(+(rho*match(s,h(theta)))^2/2)
where rho is the desired snr.
"""

__author__ = "Michael Puerrer, Prayush Kumar"
__copyright__ = "Copyright 2015"
__email__ = "Michael.Puerrer@ligo.org"

import lal, lalsimulation
import numpy as np
import sys,os
import emcee
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('text', usetex=True)
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
import triangle
from optparse import OptionParser
from scipy.stats import norm
import scipy.interpolate as ip

from match import *
from injection import *


################################################################################
# Function to compute percentile intervals from samples
################################################################################
def PercentileInterval(s, pc=95.0):
    """
    Function to compute percentile intervals from samples
    """
    #{{{
    offset = (100.0-pc)/2.0
    lowbound = s > np.percentile(s, offset)
    hibound = s < np.percentile(s, 100.0 - offset)
    mask = np.logical_and(lowbound, hibound)
    return s[mask]
    #}}}

################################################################################
# Parameter transformation functions
################################################################################
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


#################################################################################
## Waveforms' analyses functions
#################################################################################
#def ComputeMatch(theta, s):
  ## signal s
  #[ eta, Mc, chi1, chi2 ] = theta
  #q = qfun(eta)
  #M = Mfun(Mc, eta)
  #m1 = M*1.0/(1.0+q)
  #m2 = M*q/(1.0+q)
  #m1_SI = m1*lal.MSUN_SI
  #m2_SI = m2*lal.MSUN_SI
  ## print m1, m2, chi1, chi2, tc, phic
  ## generate wf
  #[hp, hc] = lalsimulation.SimInspiralChooseFDWaveform(phi0, deltaF,\
                      #m1_SI, m2_SI,\
                      #s1x, s1y, chi1, s2x, s2y, chi2,\
                      #f_min, f_max, f_ref,\
                      #distance, inclination,\
                      #0, 0, None, None, ampOrder, phOrder,\
                      #template_approximant)
  #psdfun = lalsim_psd
  #return match_FS(s, hp, psdfun, zpf=2)


#def lnprior(theta):
  #[ eta, Mc, chi2, Lambda ] = theta
  #if eta > 0.25 or eta < eta_min:
    #return -np.inf
  #q = qfun(eta)
  #M = Mfun(Mc, eta)
  #m1 = M*1.0/(1.0+q)
  #m2 = M*q/(1.0+q)
  #if m1 < m1_min or m1 > m1_max:
    #return -np.inf
  #if m2 < m2_min or m2 > m2_max:
    #return -np.inf
  #if M > options.Mtot_max:
    #return -np.inf
  #if chi2 < chi2_min or chi2 > chi2_max:
    #return -np.inf
  
  ## Additional priors to avoid calling tidal model outside of region of validity
  #if eta < 6./49.:
    #return -np.inf
  #if chi2 > 0.75 or chi2 < -0.75:
    #return -np.inf
  #if Lambda < 0 or Lambda > Lambda_max:
    #return -np.inf
  #return 0.0

#def lnlikeMatch(theta, s):
  ## signal s
  #[ eta, Mc, chi2, Lambda ] = theta
  #if eta > 0.25 or eta < eta_min:
    #return -np.inf
  #q = qfun(eta)
  #M = Mfun(Mc, eta)
  #m1 = M*1.0/(1.0+q)
  #m2 = M*q/(1.0+q)
  #m1_SI = m1*lal.MSUN_SI
  #m2_SI = m2*lal.MSUN_SI
  ## print M, q, chi1, chi2
  ## generate wf
  
  ## LAL FD waveform with tidal corrections
  #[hp, hc] = tw.getWaveform( M, eta, chi2, Lambda=Lambda, delta_f=deltaF, f_lower=f_min, f_final=f_max )
  #hp = convert_FrequencySeries_to_lalREAL16FrequencySeries( hp ) # converts from pycbc.types.frequencyseries.FrequencySeries to COMPLEX16FrequencySeries
    
  #psdfun = lalsim_psd
  #ma = match_FS(s, hp, psdfun, zpf=2)
  #if np.isnan(ma):
    #print theta, ma
    #print hp.data.data
  #rho = SNR # use global variable for now
  #return 0.5*(rho*ma)**2

#def lnprobMatch(theta, s):
  #lp = lnprior(theta)
  #if not np.isfinite(lp):
    #return -np.inf
  #return lp + lnlikeMatch(theta, s)


################################################################################
#### # Function to remove duplicate entries from a list
################################################################################
def f7(seq):
    """
    Function to remove duplicate entries from a list
    """
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x))]
    
    
################################################################################
# Combine numpy pickles
################################################################################
def combine_pickles(all_data_files, axis=1, verbose=False):
    """
 Function to combine information from similar shaped numpy pickles
 
 Assume two things:
 1. initial runs can have files that store data cumulatively. Therefore, the 
    no of samples in the files will increase linearly.
 2. Later continuation of the same run will store equal number of samples in 
    each file. 
    
 Therefore, we look at the transition point, and start accumulating samples from
 there.
 
 NOTE: The order of files passed is IMPORTANT..!
    """
    #{{{
    if verbose: print "combining ", all_data_files
    all_pickles = [np.load(fnam) for fnam in all_data_files]
    len_pickles = [np.shape(p)[1] for p in all_pickles]
    mask = (np.roll(len_pickles,-1) - len_pickles)[:] <= 0
    mask[-1] = True # always need to include the most recent data file
    if verbose: print "mask = ", mask
    tbinc_pickles = all_pickles#[all_pickles[idx] for idx in range(len(mask)) if mask[idx]]
    #
    pid = 0
    for pickle in tbinc_pickles:
      if verbose: print "shape of pickle = ", np.shape(pickle)
      if not mask[pid]: 
        if verbose: print "ignoring.."
        pid += 1
        continue
      #if np.shape(np.shape(pickle))[0] != 3:
      #  if verbose: 
      #    print >>sys.stdout, "Skipping subfile %s" % all_data_files[pid]
      #  continue
      if pid == 0:
        combined = pickle
      else: combined = np.append( combined, pickle, axis=axis )
      if verbose: print "shape of pickle now = ", np.shape(combined)
      pid += 1
    #
    return combined
    #}}}


################################################################################
## Function to load samples from partial files
################################################################################
def read_run_part_names(dataDir, SNR, burnin=500, useMaxNRun=True, \
                      chain_number=-1, return_ids=False, verbose=True):
    """
This function reads the MCMC chain and loglikelihood values from partial 
result files and combines them

(In reality it really reads the most recent file as samples are cumulatively 
  stored)
  
 Assumptions:
 1. New chains number files with new random integers, ie chain%d
 2. The second index chain%d-%d.npy is chronological, and therefore all 
    files are to be read in order, and finally combined
    
 Inputs:
 dataDir : full path to the directory where the chain/loglike.npy files exist
 SNR : signal-to-noise ratio used for the injection
 burnin : No of initial samples to be ignored from the MCMC chains
 useMaxNRRun : if True, the chain with maximum samples is used
 chain_number : if useMaxNRRun is False, this is the number corresponding to
                a particular chain
    """
    #{{{
    import glob
    # Figure out the random integer associated with each run
    all_files = glob.glob(dataDir + "/chain*-*.npy")
    if len(all_files) == 0:
      if verbose: print "No npy file found."
      if return_ids: return [], [], -1, -1
      else: return [], []
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
    # Which run to use? Assume each part has the same number of samples
    if useMaxNRun:
      last_len, last_chnum = -1, -1
      for chnum in part_files:
        if len(part_files[chnum]) > last_len:
          last_len = len(part_files[chnum])
          last_chnum = chnum
    else: 
      last_chnum, last_len = chain_number, len(part_files[chain_number])
    if verbose: 
      print >>sys.stdout, "chosen run %d, %d subfiles" % (last_chnum, last_len)
      print >>sys.stdout, "out of ", chain_nums
    # If no runs are found
    if last_len <= 0 or last_chnum <= 0:
      if return_ids: return [], [], -1, -1
      else: return [], []
    # Now open each continuation file for the chosen run and gather samples
    #print part_files[last_chnum]
    part_files_loglike = [fnam.replace('chain','loglike') for fnam in part_files[last_chnum]]    
    chain = combine_pickles( part_files[last_chnum], axis=1, verbose=verbose )
    loglike = combine_pickles( part_files_loglike, axis=0, verbose=verbose )
    #
    if verbose:
      print >>sys.stdout, \
        "reading from %s/%s" % (part_files[last_chnum][last_len-1],\
                                part_files[last_chnum][last_len-1])
      print >>sys.stdout, \
        "samples (chain) = ", np.shape(chain), " (loglike) = ", np.shape(loglike)
    
    if not return_ids: return chain, loglike
    else: return chain, loglike, last_chnum, last_len
    #}}}


################################################################################
###### Join MCMC sample data
################################################################################
def load_samples_join(dataDir, SNR, burnin=500, useMaxNRun=True, \
                      chain_number=-1, verbose=True):
    """
 This function 
 1. loads in the MCMC samples from partial files and combines them.
 2. throws away samples before burnin and at very low likelihood.
 3. convert log(L) to match.

 Assumptions:
 1. New chains number files with new random integers, ie chain%d
 2. The second index chain%d-%d.npy is chronological, and therefore all 
    files are to be read in order, and finally combined

 Inputs:
 dataDir : full path to the directory where the chain/loglike.npy files exist
 SNR : signal-to-noise ratio used for the injection
 burnin : No of initial samples to be ignored from the MCMC chains
 useMaxNRRun : if True, the chain with maximum samples is used
 chain_number : if useMaxNRRun is False, this is the number corresponding to
                a particular chain

 Option to combine samples from different runs as well
    """
    #{{{
    import glob
    chain, loglike = read_run_part_names(dataDir, SNR, burnin=burnin, \
            useMaxNRun=useMaxNRun, chain_number=chain_number, verbose=verbose )
    
    print "datadir = ", dataDir, " SNR = ", SNR
    #print chain, loglike
    if np.shape(chain)[0] != np.shape(loglike)[1] or np.shape(chain)[1] != np.shape(loglike)[0]:
      raise RuntimeError("Chain and loglikelihood file not the same length")
    if len(chain) == 0 or len(loglike) == 0:
      if verbose:
        print "Empty lists gotten for chain and loglike for %s" % dataDir
      return {'eta' : None, 'Mc' : None, 'chieff' : None, 'chiPN' : None,
              'chi1' : None, 'chi2' : None, 'Mtot' : None,
              'm1' : None, 'm2' : None, 'q' : None, 'match' : None}
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
            'm1' : m1val, 'm2' : m2val, 'q' : qval, 'match' : matchval}
    #}}}


################################################################################
# Old function, similar to load_samples_join, that does not combine samples 
# across partial files 
################################################################################
def load_samples(dataDir, SNR, burnin=500):
    """
    Load samples from double-spin emcee run. Throw away samples before burnin 
    and at very low likelihood. Convert to match.
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



