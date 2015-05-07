#
# Tools for analysis of MCMC samples from emcee for spin estimation project
#
# MP 04/2015

import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

from pydoc import help
from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde
from scipy import stats

# Constants
quantiles_68 = [0.16, 0.5, 0.84] # 1-sigma
quantiles_95 = [0.0228, 0.5, 0.9772] # 2-sigma ~ 95.4%
percentiles_68 = 100*np.array(quantiles_68)
percentiles_95 = 100*np.array(quantiles_95)
perc_int_95 = [2.28, 97.72]
perc_int_99_7 = [0.13, 99.87] # 3 sigma

# Figure settings
ppi=72.0
aspect=0.75
size=4.0 # was 6
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
    q = qfun(eta)
    M = 1
    m1 = m1fun(M,q)
    m2 = m2fun(M,q) # m2 >= m1 as in python code
    chi_s = (chi1 + chi2)/2.0;
    chi_a = (chi1 - chi2)/2.0;
    delta = (m1-m2)/M;
    return chi_s + delta * chi_a

def chi_PN(eta, chi1, chi2):
    q = qfun(eta)
    M = 1
    m1 = m1fun(M,q)
    m2 = m2fun(M,q) # m2 >= m1 as in python code
    chi_s = (chi1 + chi2)/2.0;
    chi_a = (chi1 - chi2)/2.0;
    delta = (m1-m2)/M;
    return chi_s * (1.0 - 76.0/113.0 * eta) + delta * chi_a # Ajith's PN reduced spin parameter

def load_samples(dataDir, SNR, burnin=500):
    """
    Load samples from double-spin emcee run. Throw away samples before burnin and at very low likelihood. Convert to match.
    """
    chain = np.load(dataDir+'chain.npy')
    loglike = np.load(dataDir+'loglike.npy')
    
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


def PercentileInterval(s, pc=95.0):
    offset = (100.0-pc)/2.0
    lowbound = s > np.percentile(s, offset)
    hibound = s < np.percentile(s, 100.0 - offset)
    mask = np.logical_and(lowbound, hibound)
    return s[mask]

# hist(PercentileInterval(s1[0], pc=99.0), 100);

# simple example

prefix = '/Users/mpuer/Documents/BBHSIM/const_chi_eob/Michael_emcee/match-marg-comparison/'
test_dir = prefix+'match-test/'

M = 100
SNR = 30

match_s9_q4_30_LM = {}
match_s9_q4_30_LM['samples'] = load_samples(test_dir, SNR)

eta_inj = 0.16
S = match_s9_q4_30_LM['samples']['eta']
bias = np.median(S) - eta_inj

CI90 = [np.percentile(S, 5), np.percentile(S, 95)]
width90 = CI90[1] - CI90[0]

print bias / width90
