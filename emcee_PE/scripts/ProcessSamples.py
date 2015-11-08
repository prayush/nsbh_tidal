#!/usr/bin/env python
# Tools for analysis of MCMC samples from emcee for spin estimation project
#
# MP 04/2015, PK 2015
import os, sys
import commands as cmd
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

def calculate_bias(\
  basedir='/home/prayush/projects/nsbh/TidalParameterEstimation/ParameterBiasVsSnr/SEOBNRv2/set005/TN/',\
  simdir='TN_q2.00_mNS1.35_chiBH0.50_Lambda500.0_SNR60.0/NW100_NS6000/',\
  M_inj = 3*1.35, eta_inj = 2./9., chi1_inj=0, chi2_inj=0.5, Lambda_inj=0, SNR_inj = 60,\
  biastype='fractional', recover_tidal=False, \
  confidence_levels=[90.0, 68.26895, 95.44997, 99.73002]):
    """
    Load samples for a given physical configuration, 
    1. decode the location of the corresponding run. 
    2. Compute 68%, 90%,.. confidence interval for different sampled parameters
    3. Compute the median value of different parameters fromt the posterior.
    
    We plan to store 5 quantities for each parameter k:
    k+0: Median recovered parameter value from posterior samples
    k+1: Bias = (X(median) - X(inj))/X(inj)
    k+2: Confidence level lower bound = X(confidence_level_low)
    k+3: Confidence level upper bound = X(confidence_level_up)
    k+4: Confidence interval = (X(confidence_level_up) - X(confidence_level_low))/X(inj)
    
    PLUS
    0 : Confidence interval probability
    1 : max LogLikelihood value
    
    """
    #{{{
    test_dir = os.path.join(basedir, simdir)
    m1_inj, m2_inj = pnutils.mchirp_eta_to_mass1_mass2(M_inj * eta_inj**0.6, eta_inj)
    params_inj = {'eta' : eta_inj, 'Mtot' : M_inj, 'Mc' : M_inj * eta_inj**0.6,
                  'chi1' : chi1_inj, 'chi2' : chi2_inj, 'Lambda' : Lambda_inj,
                  'm1' : m1_inj, 'm2' : m2_inj}
    
    match = {}
    match['samples'] = load_samples_join(test_dir, SNR_inj)
    
    if recover_tidal:
      match['samples']['Lambda'] = match['samples']['chi2']
      match['samples']['chi2']   = match['samples']['chi1']
      parameters = ['m1', 'm2', 'Mc', 'Mtot', 'eta', 'chi2', 'Lambda']
    else:
      parameters = ['m1', 'm2', 'Mc', 'Mtot', 'eta', 'chi2']
    
    num_of_data_fields = 5
    summary_data = np.zeros(( len(confidence_levels), len(parameters)*num_of_data_fields + 1 + 1 ))
    
    # Populate first column of summary data : confidence level probabilities
    for idx in range(len(confidence_levels)):
      summary_data[idx,0] = confidence_levels[idx] / 100.
      summary_data[idx,1] = np.max( match['samples']['match'] ) 
    
    idx = 2
    for param in parameters:
      # get the posterior samples
      S = match['samples'][param]
      if S == None:
        raise RuntimeError("Could not find samples for parameter %s" % param)
      #
      param_inj = params_inj[param]
      for jdx, confidence_level in enumerate(confidence_levels):
        llimit = (100. - confidence_level) / 2.
        ulimit = 100. - llimit
        
        # Fractional bias
        summary_data[jdx, idx] = np.median(S)
        summary_data[jdx, idx+1] = np.median(S) - param_inj
                
        # Confidence interval limits
        summary_data[jdx, idx+2] = np.percentile(S, llimit)
        summary_data[jdx, idx+3] = np.percentile(S, ulimit)
        
        # Confidence interval width as a fraction of parameter value
        summary_data[jdx, idx+4] = summary_data[jdx, idx+3] - summary_data[jdx, idx+2]
        
        # Normalize        
        if 'chi' not in param and 'fractional' in biastype and param_inj != 0:
          summary_data[jdx, idx+1] /= param_inj
          summary_data[jdx, idx+4] /= param_inj            
      #  
      idx += num_of_data_fields
    #
    return summary_data
    #}}}



def calculate_store_biases(qvec=None, chi2vec=None, Lambdavec=None, SNRvec=None,\
            Nsamples=[150000], Nwalkers=[100], outfile='output.h5', mNS = 1.35,\
            recover_tidal=True):
  #{{{
  ''' 
  This function loops over all parameters for which ranges are given as input,
  reads the posterior samples for each injection combination, and computes 
  biases in different parameters' recovery for each injected parameter 
  combination.  
  '''
  fout = h5py.File(outfile, 'w')
  
  for qidx, q in enumerate(qvec):
    grp_l1 = 'q%.1f.dir' % q
    fout.create_group(grp_l1)
    for x2idx, chi2 in enumerate(chi2vec):
      grp_l2 = 'chiBH%.2f.dir' % chi2
      if grp_l2 not in fout[grp_l1].keys(): fout[grp_l1].create_group(grp_l2)
      for Lidx, Lambda in enumerate(Lambdavec):
        grp_l3 = 'LambdaNS%.1f.dir' % Lambda
        if grp_l3 not in fout[grp_l1][grp_l2].keys():
          fout[grp_l1][grp_l2].create_group(grp_l3)
        for SNR in SNRvec:
          dset_l1 = 'SNR%.1f.dat' % SNR
          for Ns in Nsamples:
            for Nw in Nwalkers:
              simdir = mr.get_simdirname(q, mNS, chi2, Lambda, SNR, Nw, Ns)
              print "\n\n Trying to read in %s" % simdir
              #
              summ_data = calculate_bias(basedir=cmd.getoutput('pwd -P'),\
                    simdir=simstring+simdir+'/',\
                    M_inj=(1.+q)*mNS, eta_inj=q/(1.+q)**2,\
                    chi1_inj=chi1, chi2_inj=chi2, Lambda_inj=Lambda,\
                    SNR_inj=SNR, recover_tidal=recover_tidal)
              fout[grp_l1][grp_l2][grp_l3].create_dataset(dset_l1, data=summ_data)
              
  fout.close()
  #}}}


######################################################
# Set up parameters of signal
######################################################
chi1 = 0.   # small BH
chi2vec = [-0.5, 0, 0.5, 0.74999]  # larger BH
mNS = 1.35
qvec = [2, 3, 4]
Lambdavec = [0]
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

######################################################
# Set up RUN parameters
######################################################
if inject_tidal: sigstring = 'T'
else: sigstring = 'N'
if recover_tidal: tmpstring = 'T'
else: tmpstring = 'N'
simstring = sigstring + tmpstring + '_'

if inject_tidal: Lambdavec = [500, 800, 1000]

Nwalkers = [100]
Nsamples = [150000]
Nburnin  = 500

#qvec = [2]
#chi2vec = [-0.5, 0.5]
#Lambdavec = [0]
#SNRvec = [30, 50]

print "\n\n\n"
print "Storing data in HDF file for ", qvec, chi2vec, Lambdavec, SNRvec
print "\n\n\n"
calculate_store_biases(qvec=qvec, chi2vec=chi2vec, Lambdavec=Lambdavec,\
      SNRvec=SNRvec, recover_tidal=recover_tidal,\
      outfile=simstring+'ParameterBiasesAndConfidenceIntervals.h5')
