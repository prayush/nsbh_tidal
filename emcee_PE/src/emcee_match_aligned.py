#!/usr/bin/env python

# -*- coding: utf-8 -*-
""" MCMC python code using emcee to compute 4D credible regions
This version uses parameters [eta, Mc, chi1, chi2] for SEOBNRv2 ROM templates.

The idea is to use a match-based (zero-noise) likelihood 
L(s, theta) = exp(+(rho*match(s,h(theta)))^2/2)
where rho is the desired snr.
"""

__author__ = "Michael Puerrer"
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
import triangle, corner
from optparse import OptionParser
from scipy.stats import norm
import scipy.interpolate as ip
import random

from match import *
from injection import *
from utils import *

parser = OptionParser()
parser.add_option("-n", "--nsamples", dest="nsamples", type="int",
                  help="How many samples to produce per walker", metavar="nsamples")
parser.add_option("-w", "--nwalkers", dest="nwalkers", type="int",
                  help="How many walkers to use.", metavar="nwalkers")
parser.add_option("-q", "--q_signal", dest="q_signal", type="float",
                  help="Asymmetric mass-ratio of signal (q>=1).", metavar="q_signal")
parser.add_option("-M", "--M_signal", dest="M_signal", type="float",
                  help="Total mass of signal in Msun.", metavar="M_signal")
parser.add_option("-c", "--chi1_signal", dest="chi1_signal", type="float",
                  help="Dimensionless spin chi_1 of smaller body of signal.", metavar="chi1_signal")
parser.add_option("-d", "--chi2_signal", dest="chi2_signal", type="float",
                  help="Dimensionless spin chi_2 of larger body of signal.", metavar="chi2_signal")
parser.add_option("-l", "--f_min", dest="f_min", type="float",
                  help="Lower cutoff frequency for match [Hz].", metavar="f_min")
parser.add_option("-u", "--f_max", dest="f_max", type="float",
                  help="Upper cutoff frequency for match [Hz].", metavar="f_max")
parser.add_option("-f", "--deltaF", dest="deltaF", type="float",
                  help="Lower cutoff frequency for match [Hz].", metavar="deltaF")
parser.add_option("-a", "--m1_min", dest="m1_min", type="float",
                  help="Lower prior boundary for component mass 1. m2 >= m1", metavar="m1_min")
parser.add_option("-b", "--m1_max", dest="m1_max", type="float",
                  help="Upper prior boundary for component mass 1. m2 >= m1", metavar="m1_max")
parser.add_option("-e", "--m2_min", dest="m2_min", type="float",
                  help="Lower prior boundary for component mass 2. m2 >= m1", metavar="m2_min")
parser.add_option("-g", "--m2_max", dest="m2_max", type="float",
                  help="Upper prior boundary for component mass 2. m2 >= m1", metavar="m2_max")
parser.add_option("-m", "--Mtot_max", dest="Mtot_max", type="float",
                  help="Upper prior boundary for total mass", metavar="Mtot_max")
parser.add_option("-A", "--chi1_min", dest="chi1_min", type="float",
                  help="Lower prior boundary for spin1 of signal.", metavar="chi1_min")
parser.add_option("-B", "--chi1_max", dest="chi1_max", type="float",
                  help="Upper prior boundary for spin2 of signal.", metavar="chi1_max")
parser.add_option("-C", "--chi2_min", dest="chi2_min", type="float",
                  help="Lower prior boundary for spin2 of signal.", metavar="chi2_min")
parser.add_option("-D", "--chi2_max", dest="chi2_max", type="float",
                  help="Upper prior boundary for spin2 of signal.", metavar="chi2_max")
parser.add_option("-E", "--eta_min", dest="eta_min", type="float",
                  help="Lower prior boundary for symmetric mass-ratio", metavar="eta_min")
parser.add_option("-s", "--snr", dest="SNR", type="float",
                  help="Signal SNR.", metavar="SNR")
parser.add_option("-r", "--burnin", dest="burnin", type="int",
                  help="How many samples to discard at the beginning of each chain.", metavar="burnin")
parser.add_option("--burnend", dest="burnend", type="int",
                  help="How many samples to discard at the end of each chain.", metavar="burnend")
parser.add_option("-S", "--signal_approximant", dest="signal_approximant", type="string",
                  help="Which approximant to use as a signal.", metavar="signal_approximant")
parser.add_option("-T", "--template_approximant", dest="template_approximant", type="string",
                  help="Which (FD) approximant to use as templates.", metavar="template_approximant")
parser.add_option("-P", "--psd", dest="lalsim_psd", type="string",
                  help="Which lalsimulation power spectral density to use.", metavar="lalsim_psd")
parser.add_option("-V", "--eta_stdev_init", dest="eta_stdev_init", type="float",
                  help="Standard deviation for initial walker configurations in symmetric mass-ratio eta.", metavar="eta_stdev_init")
parser.add_option("-X", "--Mc_stdev_init", dest="Mc_stdev_init", type="float",
                  help="Standard deviation for initial walker configurations in chirp mass Mc.", metavar="Mc_stdev_init")
parser.add_option("-Y", "--chi1_stdev_init", dest="chi1_stdev_init", type="float",
                  help="Standard deviation for initial walker configurations in chi1.", metavar="chi1_stdev_init")
parser.add_option("--chi2_only", dest="chi2_only", action="store_true",
                  help="Flag to be set to indicate that only chi2 - larger objects spin - is to be sampled", 
                  metavar="chi2_only")
parser.add_option("-Z", "--chi2_stdev_init", dest="chi2_stdev_init", type="float",
                  help="Standard deviation for initial walker configurations in chi2.", metavar="chi2_stdev_init")
parser.add_option("-R", "--post-process", dest="post_process", type="string",
                  help='Only run postprocessing and read chain and loglikelihood from specified directory.', metavar="post_process_from_directory")
parser.add_option("-J", "--signal-from-datafile", dest="signal_from_datafile", type="string",
                  help="Read injection from ASCII datafile specified here.", metavar="signal_from_datafile")
parser.add_option("-p", "--parallel-tempering", dest="pt", action="store_true",
                  help="Use parallel tempering ensemble sampler.", metavar="parallel_tempering")
parser.add_option("--no-parallel-tempering", dest="pt", action="store_false",
                  help="Don't use parallel tempering ensemble sampler.", metavar="parallel_tempering")
parser.add_option("-t", "--number-of-threads", dest="nThreads", type="int",
                  help="Number of threads for ensemble sampler.", metavar="number_of_threads")
parser.add_option("-U", "--resume", dest="resume", type="string",
                  help="Restart run from chain.npy and loglike.npy from specified directory. Make sure the commandline parameters agree with the produced samples!", metavar="resume")
parser.add_option("--auto-resume", dest="auto_resume", action="store_false", \
            help="Restart the chain that has collected maximum steps",\
            metavar="auto_resume")
parser.add_option("--verbose", action="store_true", default=False)
parser.add_option("--debug", action="store_true", default=False)

# Additional options for tidal corection waveforms
parser.add_option("", "--inject-tidal", dest="inject_tidal", action="store_true",
                  help="Inject waveform LAL waveform with additional tidal corrections.", metavar="inject_tidal")
parser.add_option("", "--recover-tidal", dest="recover_tidal", action="store_true",
                  help="Recover with waveform LAL waveform with additional tidal corrections.", metavar="recover_tidal")
parser.add_option("-L", "--Lambda_signal", dest="Lambda_signal", type="float",
                  help="Tidal deformability Lambda of signal.", metavar="Lambda_signal")

parser.set_defaults(nsamples=1000, nwalkers=100, q_signal=4.0, M_signal=100.0, 
  chi1_signal=0.0, chi2_signal=0.0, f_min=10, f_max=2048, deltaF=0.5, 
  m1_min=6.0, m1_max=100.0, m2_min=6.0, m2_max=300.0, Mtot_max=500.0, SNR=15, burnin=250, burnend=0, eta_min=0.01,
  signal_approximant='lalsimulation.SEOBNRv2_ROM_DoubleSpin', 
  template_approximant='lalsimulation.SEOBNRv2_ROM_DoubleSpin',
  lalsim_psd='lalsimulation.SimNoisePSDaLIGOZeroDetHighPower',
  chi2_only=False,
  chi1_min=-1, chi1_max=0.99, chi2_min=-1, chi2_max=0.99,
  eta_stdev_init=0.15, Mc_stdev_init=5, chi1_stdev_init=0.4, chi2_stdev_init=0.4, post_process='',
  nThreads=1, pt=False, resume='', auto_resume=True, # Hard set auto-resume
  inject_tidal=False, recover_tidal=False, Lambda_signal=500)

(options, args) = parser.parse_args()

post_process_directory = options.post_process
if post_process_directory != '':
  post_process = True
else:
  post_process = False

resume_directory = options.resume
if resume_directory != '':
  resume = True
else:
  resume = False
auto_resume = options.auto_resume 

if resume and post_process:
  print 'Resume and post_process options are exclusive! Aborting.'
  sys.exit(-1)

pt = options.pt
nThreads = options.nThreads # LAL calls fail with more than 1 thread. Why?

eta_stdev_init = options.eta_stdev_init
Mc_stdev_init = options.Mc_stdev_init
chi1_stdev_init = options.chi1_stdev_init
chi2_stdev_init = options.chi2_stdev_init

if Mc_stdev_init < 1e-6:
  print 'Mc_stdev_init < 1e-6!'
  print 'Setting Mc_stdev_init = 0.0001'
  Mc_stdev_init = 1e-4

if options.M_signal < options.m1_min + options.m2_min or options.M_signal > options.m1_max + options.m2_max:
  print 'Error: M_signal should be inside the mass prior [m1_min+m2_min, m1_max+m2_max]'
  sys.exit(-1)

# Parameter settings
nsamples = options.nsamples
nwalkers = options.nwalkers
burnin = options.burnin 
burnend= options.burnend

q_true = options.q_signal
M_true = options.M_signal
chi1_true = options.chi1_signal
chi2_true = options.chi2_signal
f_min = options.f_min
f_max = options.f_max
deltaF = options.deltaF
SNR = options.SNR

eta_true = etafun(q_true)
Mc_true = Mchirpfun(M_true, eta_true)

phi0 = 0
m1 = M_true * 1.0/(1.0+q_true)
m2 = M_true * q_true/(1.0+q_true)
m1_SI = m1 * lal.MSUN_SI
m2_SI = m2 * lal.MSUN_SI
dist = 10.  # Mpc
f_ref = 100.
inclination = 0. * np.pi
s1x = 0.
s1y = 0.
s1z = chi1_true
s2x = 0.
s2y = 0.
s2z = chi2_true
ampOrder = -1
phOrder = -1
distance = dist * 1.e6 * lal.PC_SI

# specify prior ranges
m1_min, m1_max = options.m1_min, options.m1_max
m2_min, m2_max = options.m2_min, options.m2_max
chi1_min, chi1_max = options.chi1_min, options.chi1_max
chi2_min, chi2_max = options.chi2_min, options.chi2_max
eta_min = options.eta_min
eta_max = 0.25
Mc_min = (m1_min+m2_min)*eta_min**(3.0/5.0)
Mc_max = (m1_max+m2_max)*eta_max**(3.0/5.0)

signal_approximant=eval(options.signal_approximant)
template_approximant=eval(options.template_approximant)

inject_tidal = options.inject_tidal
recover_tidal = options.recover_tidal
Lambda_signal = options.Lambda_signal

# FIXME: add a general psd from file option
if options.lalsim_psd == 'ET-D': # not in lalsimulation
  n = int(options.f_max / options.deltaF)
  ET_psd = lal.CreateREAL8FrequencySeries('ET_psd', 0, 1, options.deltaF, 1, n)
  lalsimulation.SimNoisePSDFromFile(ET_psd, 1, '/Users/mpuer/Downloads/ET.txt') # FIXME need an option for path to file
  ET_freq = arange(n)*deltaF
  mask = ET_psd.data.data > 0
  ET_D_psd = ip.InterpolatedUnivariateSpline(ET_freq[mask], ET_psd.data.data[mask])
  # even though this evaluates to an array for a single number, it should work with the match functions
else:
  lalsim_psd=eval(options.lalsim_psd)

post_process_directory = options.post_process
if post_process_directory != '':
  post_process = True
else:
  post_process = False

# we just use hp here; need to change only if we want to inject precessing approximants


# Generate injection
if options.signal_from_datafile != None:
  print "Reading injection from file", options.signal_from_datafile
  # Expect positive frequency data for [f, rehp, imhp, rehc, imhc] in a text file.
  # The data has to follow the lalsimulation convention and start at f=0 Hz.
  [f, rehp, imhp, rehc, imhc] = np.loadtxt(options.signal_from_datafile).T
  if f[0] != 0.0:
    print "Injection data has to follow lalsimulation convention and start at f=0 Hz!"
    sys.exit(-1)
  print "Warning: Injection from file code has not been tested!"
  # FIXME: Will need to reinterpolate strain onto correct frequencies! Decompose into
  # amplitude and phase; or to avoid this make sure that the data has the
  # correct deltaF!
  if np.max(np.abs(np.diff(f) - deltaF)) > deltaF*0.001:
    print "Frequency spacing of injection and deltaF disagree. Please fix it."
    sys.exit(-1)
  hps = rehp + 1j*imhp
  hcs = rehc + 1j*imhc
else:
  print "Using lalsimulation FD injection"
  try:
    print "Preparing injection for ", options.signal_approximant
    if lalsim.SimInspiralImplementedFDApproximants(signal_approximant): 
      # FD approximant : call wrapper around ChooseFDWaveform()
      if inject_tidal:
        print 'Injecting FD approximant with tidal modifications based on ChooseFDWaveform()'
        [hps, hcs] = InjectTidalWaveform_ChooseFD(m1, m2,\
                        S1x=s1x, S1y=s1y, S1z=s1z, S2x=s2x, S2y=s2y, S2z=s2z,\
                        Lambda=Lambda_signal,\
                        f_min=f_min, f_max=f_max, f_ref=f_ref, deltaF=deltaF,\
                        approximant=signal_approximant,\
                        make_plots=True)
      else:
        print 'Injecting FD approximant via ChooseFDWaveform()'
        [hps, hcs] = InjectWaveform_ChooseFD(m1, m2,\
                        S1x=s1x, S1y=s1y, S1z=s1z, S2x=s2x, S2y=s2y, S2z=s2z,\
                        f_min=f_min, f_max=f_max, f_ref=f_ref, deltaF=deltaF,\
                        approximant=signal_approximant,\
                        make_plots=True)
    else: 
      # TD approximant : InjectWaveform will call SimInspiralTD()
      print 'Injecting TD approximant via SimInspiralTD()'
      [hps, hcs] = InjectWaveform(m1, m2,\
                        S1x=s1x, S1y=s1y, S1z=s1z, S2x=s2x, S2y=s2y, S2z=s2z,\
                        f_min=f_min, f_max=f_max, f_ref=f_ref, deltaF=deltaF,\
                        approximant=signal_approximant,\
                        make_plots=True)    
  except RuntimeError:
    print 'Failed to generate FD injection for lalsimulation approximant %s.' %(options.signal_approximant)
    raise

def ComputeMatch(theta, s):
  # signal s
  [ eta, Mc, chi1, chi2 ] = theta
  q = qfun(eta)
  M = Mfun(Mc, eta)
  m1 = M*1.0/(1.0+q)
  m2 = M*q/(1.0+q)
  m1_SI = m1*lal.MSUN_SI
  m2_SI = m2*lal.MSUN_SI
  # print m1, m2, chi1, chi2, tc, phic
  # generate wf
  [hp, hc] = lalsimulation.SimInspiralChooseFDWaveform(phi0, deltaF,\
                      m1_SI, m2_SI,\
                      s1x, s1y, chi1, s2x, s2y, chi2,\
                      f_min, f_max, f_ref,\
                      distance, inclination,\
                      0, 0, None, None, ampOrder, phOrder,\
                      template_approximant)
  psdfun = lalsim_psd
  return match_FS(s, hp, psdfun, zpf=2)

print [eta_true, Mc_true, chi1_true, chi2_true]
print 'match (s, h^*)', ComputeMatch([eta_true, Mc_true, chi1_true, chi2_true], hps)


print "nsamples: {0:4d} nwalkers: {1:4d}\n".format(nsamples, nwalkers)
print "Signal parameters: (m2>=m1)"
print "q    = ", q_true
print "M    = ", M_true
print "m1   = ", m1
print "m2   = ", m2
print "chi1 = ", chi1_true
print "chi2 = ", chi2_true
print "f_min: {0:g} f_max: {1:g} deltaF: {2:g}\n".format(f_min, f_max, deltaF)
print "Prior:\n [m1_min, m1_max] = [{0}, {1}], [m2_min, m2_max] = [{2}, {3}]".format(m1_min, m1_max, m2_min, m2_max)
print "Prior:\n [eta_min, eta_max] = [{0}, {1}], [chi1_min, chi1_max] = [{2}, {3}], [chi1_min, chi1_max] = [{4}, {5}], Mtot_max = {6}".format(eta_min, eta_max, chi1_min, chi1_max, chi2_min, chi2_max, options.Mtot_max)
print "Signal snr: ", SNR
print "Using {0} as signal and {1} as templates.".format(options.signal_approximant, options.template_approximant)
print "Using PSD ", options.lalsim_psd
if inject_tidal:
  print "Lambda_signal = ", Lambda_signal

if recover_tidal:
  print 'Templates are LAL FD waveform with tidal corrections'
  # Instantiate the class only once
  tw = tidalWavs(approx=options.template_approximant.split('.')[1], verbose=False)

# Define the probability function as likelihood * prior.
ndim = 4
def lnprior(theta):
  [ eta, Mc, chi1, chi2 ] = theta
  if eta > 0.25 or eta < eta_min:
    return -np.inf
  q = qfun(eta)
  M = Mfun(Mc, eta)
  m1 = M*1.0/(1.0+q)
  m2 = M*q/(1.0+q)
  #
  if options.debug:
    print "In LNPRIOR: [eta, Mc, chi1, chi2] = ", theta, " eta_min = %.2f, m1_max = %.2f, m2_max = %.2f" % (eta_min, m1_max, m2_max)
    print "In LNPRIOR  [q, M, m1, m2] = ", [q, M, m1, m2], " m1 in [%.2f, %.2f], m2 in [%.2f, %.2f], chi1 in [%.2f, %.2f], chi2 in [%.2f, %.2f]" % (m1_min, m1_max, m2_min, m2_max, chi1_min, chi1_max, chi2_min, chi2_max)
  #
  if m1 < m1_min or m1 > m1_max:
    return -np.inf
  if m2 < m2_min or m2 > m2_max:
    return -np.inf
  if M > options.Mtot_max:
    return -np.inf
  if chi1 < chi1_min or chi1 > chi1_max:
    return -np.inf
  if chi2 < chi2_min or chi2 > chi2_max:
    return -np.inf

  # Additional priors to avoid calling tidal model outside of region of validity
  if recover_tidal and eta < 6./49.:
    return -np.inf
  if recover_tidal and (chi2 > 0.75 or chi2 < -0.75):
    return -np.inf
  return 0.0

################################################################################
# Waveforms' analyses functions
################################################################################

def lnlikeMatch(theta, s):
  # signal s
  [ eta, Mc, chi1, chi2 ] = theta
  if options.chi2_only: chi1 = 0
  if eta > 0.25 or eta < eta_min:
    return -np.inf
  q = qfun(eta)
  M = Mfun(Mc, eta)
  m1 = M*1.0/(1.0+q)
  m2 = M*q/(1.0+q)
  #
  if options.debug:
    print "In LNlikeMATCH: [eta, Mc, chi1, chi2] = ", theta, " eta_min = %.2f, m1_max = %.2f, m2_max = %.2f" % (eta_min, m1_max, m2_max)
    print "In LNPRIOR:  [q, M, m1, m2] = ", [q, M, m1, m2], " m1 in [%.2f, %.2f], m2 in [%.2f, %.2f], chi1 in [%.2f, %.2f], chi2 in [%.2f, %.2f]" % (m1_min, m1_max, m2_min, m2_max, chi1_min, chi1_max, chi2_min, chi2_max)
  #
  m1_SI = m1*lal.MSUN_SI
  m2_SI = m2*lal.MSUN_SI
  # print M, q, chi1, chi2
  # generate wf

  if recover_tidal:
    # LAL FD waveform with tidal corrections
    # FIXME: For the moment we set Lambda_template = Lambda_signal
    [hp, hc] = tw.getWaveform( M, eta, chi2, Lambda=Lambda_signal, delta_f=deltaF, f_lower=f_min, f_final=f_max )
    hp = convert_FrequencySeries_to_lalREAL16FrequencySeries( hp ) # converts from pycbc.types.frequencyseries.FrequencySeries to COMPLEX16FrequencySeries
    assert hp.deltaF == deltaF
  else:
    # standard LAL FD waveform
    [hp, hc] = lalsimulation.SimInspiralChooseFDWaveform(phi0, deltaF, m1_SI, m2_SI, s1x, s1y, chi1, s2x, s2y, chi2, f_min, f_max, f_ref, distance, inclination, 0, 0, None, None, ampOrder, phOrder, template_approximant)

  psdfun = lalsim_psd
  ma = match_FS(s, hp, psdfun, zpf=2)
  if np.isnan(ma):
    print theta, ma
    print hp.data.data
  rho = SNR # use global variable for now
  return 0.5*(rho*ma)**2

def lnprobMatch(theta, s):
  lp = lnprior(theta)
  if options.debug:
    print "\n\nIn LNPROBMatch: Log(Prior) for theta = ", theta, " is ", lp
  if not np.isfinite(lp):
    return -np.inf
  post = lp + lnlikeMatch(theta, s)
  if options.debug:
    print "\n\nIn LNPROBMatch: Log(Post) for theta = ", theta, " is ", post
  #return lp + lnlikeMatch(theta, s)
  return post


# MismatchThreshold[nu_, P_, SNR_] := Quantile[ChiSquareDistribution[nu], P]/(2 SNR^2)
# (*
#   nu   : dimensionality of parameter space
#   P   : percentile for confidence region
#   SNR : desired SNR
# *)
CR_threshold_Baird_SNRs = [5, 8, 10, 15, 20, 25, 30, 35, 40, 50, 80, 100]
CR_90pc_threshold_Baird_et_al_n4 = [0.155, 0.06, 0.0388, 0.01728, 0.00972, 0.00622, 0.004321, 0.003175, 0.002431, 0.001555, 0.0006, 0.0003889]
CR_99pc_threshold_Baird_et_al_n4 = [0.265534, 0.103724, 0.0663835, 0.0295038, 0.0165959, 0.0106214, 0.00737595, 0.00541906, 0.00414897, 0.00265534, 0.00103724, 0.000663835]
CR_99p9pc_threshold_Baird_et_al_n4 = [0.369337, 0.144272, 0.0923341, 0.0410374, 0.0230835, 0.0147735, 0.0102593, 0.00753748, 0.00577088, 0.00369337, 0.00144272, 0.000923341]
CR_90pc_threshold_Baird_et_al_n4_fun = ip.InterpolatedUnivariateSpline(CR_threshold_Baird_SNRs, CR_90pc_threshold_Baird_et_al_n4)
CR_99pc_threshold_Baird_et_al_n4_fun = ip.InterpolatedUnivariateSpline(CR_threshold_Baird_SNRs, CR_99pc_threshold_Baird_et_al_n4)
CR_99p9pc_threshold_Baird_et_al_n4_fun = ip.InterpolatedUnivariateSpline(CR_threshold_Baird_SNRs, CR_99p9pc_threshold_Baird_et_al_n4)
CR_90pc_threshold = CR_90pc_threshold_Baird_et_al_n4_fun(SNR).item()
CR_99pc_threshold = CR_99pc_threshold_Baird_et_al_n4_fun(SNR).item()
CR_99p9pc_threshold = CR_99p9pc_threshold_Baird_et_al_n4_fun(SNR).item()
print "Baird et al 90% threshold", CR_90pc_threshold # For this SNR
print "Baird et al 99% threshold", CR_99pc_threshold
print "Baird et al 99.9% threshold", CR_99p9pc_threshold

if not post_process:
  # Check if auto resuming is enabled first. If it is, try to find samples.
  # If samples cannot be found, ensure that normal run starts from scratch.
  # 
  if auto_resume:
    print "Running in auto-resume mode."
    print "Loading data from directory ", resume_directory
    dataDir = '.'
    chain, loglike, chainid, partid = read_run_part_names(\
            dataDir, SNR, burnin=burnin, return_ids=True, verbose=True )
    if chainid < 0 and partid < 0:
      if True: print "Failed to auto resume."
      auto_resume = False
    else:
      print "Will write future data to %d-%d.npy" % (chainid, partid)
      print "shape of chain, loglike = ", np.shape(chain), np.shape(loglike)
      # move the original samples and log-likelihood to a backup file
      #print 'Moving ', os.path.join(resume_directory, "chain.npy"), 'to', os.path.join(resume_directory, "chain0.npy")
      #os.rename(os.path.join(resume_directory, "chain.npy"), os.path.join(resume_directory, "chain0.npy"))
      #os.rename(os.path.join(resume_directory, "loglike.npy"), os.path.join(resume_directory, "loglike0.npy"))
      #
      ## p0=chain[:,-1,:] # very simplistic: Set initial walkers up from last sample
      #
      # use all good samples from last 50 iterations and extract as many as we need to restart (nwalkers)
      k = min(50000, len(loglike[:,0]))
      print "Using good samples from last %d iterations" %(k)
      match_cut = 1 - CR_99p9pc_threshold
      loglike_ok = loglike[-k:] # use samples from last k iterations
      print "len(loglike_ok) = ", np.shape(loglike_ok)
      matches_ok = np.sqrt(2*loglike_ok)/SNR
      #if not np.any( matches_ok > match_cut ): match_cut = 0.99 # CHECKME: Arbitrary condition to enable resume 
      print "len(matches_ok) = ", len(matches_ok)
      sel = np.isfinite(loglike_ok) & (matches_ok > match_cut)
      # CHECKME: Arbitrary condition to enable resume 
      while np.count_nonzero(sel) < nwalkers:
        match_cut -= 0.001
        print " trying match cut = ", match_cut
        sel = np.isfinite(loglike_ok) & (matches_ok > match_cut)
      print "k = ", k, "\n shape and No of True elements in sel = ", np.shape(sel), np.count_nonzero(sel)
      print "shape of map(lambda i: chain[:,-k:,i].T[sel], range(ndim)) = ", np.shape(map(lambda i: chain[:,-k:,i].T[sel], range(ndim)))
      print "shape of chain[:,-k:,i] = ", np.shape(chain[:,-k:,0])
      print "shape of chain[:,-k:,i].T = ", np.shape(chain[:,-k:,0].T)
      print "shape of chain[:,-k:,i].T[sel] = ", np.shape(chain[:,-k:,0].T[sel])
      chain_ok = np.array(map(lambda i: chain[:,-k:,i].T[sel], range(ndim))) # extract all 'good' samples
      print "shape of chain_ok = ", np.shape(chain_ok), " len(chain_ok[0]) = ", len(chain_ok[0])
      idx = random.sample(xrange(len(chain_ok[0])), nwalkers) # get exactly nwalkers samples
      print "shape of idx = ", np.shape(idx)
      p0 = np.array(map(lambda i: chain_ok[:,i], idx))
      print "shape of p0 = ", np.shape(p0)
  
  if not resume and not auto_resume:
    # Setup initial walker positions
    # We could simply call sample_ball like so, but this will very likely include points outside of the prior range
    # p0 = emcee.utils.sample_ball(np.array([eta_true, Mc_true, chi1_true, chi2_true]), np.array([eta_stdev_init, Mc_stdev_init, chi1_stdev_init, chi2_stdev_init]), nwalkers)
    # may lead to warnings: ensemble.py:335: RuntimeWarning: invalid value encountered in subtract
  
    # Instead we setup walkers with sane positions drawn from a multivariate normal distribution cut at the prior boundaries
    print 'Setting up sane initial walker positions'
    p0 = []
    while len(p0) < nwalkers:
        p = emcee.utils.sample_ball(np.array([eta_true, Mc_true, chi1_true, chi2_true]), np.array([eta_stdev_init, Mc_stdev_init, chi1_stdev_init, chi2_stdev_init]), 1)[0]
        if np.isfinite(lnprobMatch(p, hps)):
            p0.append(p)
    p0 = np.array(p0)
    print 'Done setting up initial walker positions'

    # plot intitial distribution of walkers
    pl.clf()
    fig, axs = pl.subplots(2, 2, sharex=False, sharey=False, figsize=(12,8))
    axs[0,0].hist(p0.T[0], 20)
    axs[0,0].set_xlabel(r'$\eta$')
    axs[0,0].axvline(x=eta_min, c='r', ls='--')
    axs[0,0].axvline(x=eta_max, c='r', ls='--')
    axs[0,1].hist(p0.T[1], 20)
    axs[0,1].set_xlabel(r'$M_c$')
    axs[0,1].axvline(x=Mc_min, c='r', ls='--')
    axs[0,1].axvline(x=Mc_max, c='r', ls='--')
    axs[1,0].hist(p0.T[2], 20)
    axs[1,0].set_xlabel(r'$\chi_1$')
    axs[1,0].axvline(x=options.chi1_min, c='r', ls='--')
    axs[1,0].axvline(x=options.chi1_max, c='r', ls='--')
    axs[1,1].hist(p0.T[3], 20)
    axs[1,1].set_xlabel(r'$\chi_2$')
    axs[1,1].axvline(x=options.chi2_min, c='r', ls='--')
    axs[1,1].axvline(x=options.chi2_min, c='r', ls='--') 
    fig.suptitle('Starting positions for walkers', fontsize=18);
    fig.savefig('initial_walker_positions.png')
    pl.close(fig)
  elif resume and not auto_resume:
    print "Running in resume mode."
    print "Loading data from directory", resume_directory
    # NOTE: with the addition of a random number in the file names the user needs to rename the desired files to 'chain.npy' and 'loglike.npy'. Otherwise the code doesn't know which files to pick.
    chain = np.load(os.path.join(resume_directory, "chain.npy"))
    loglike = np.load(os.path.join(resume_directory, "loglike.npy"))
    
    # move the original samples and log-likelihood to a backup file
    print 'Moving ', os.path.join(resume_directory, "chain.npy"), 'to', os.path.join(resume_directory, "chain0.npy")
    os.rename(os.path.join(resume_directory, "chain.npy"), os.path.join(resume_directory, "chain0.npy"))
    os.rename(os.path.join(resume_directory, "loglike.npy"), os.path.join(resume_directory, "loglike0.npy"))
    
    # p0=chain[:,-1,:] # very simplistic: Set initial walkers up from last sample
    
    # use all good samples from last 50 iterations and extract as many as we need to restart (nwalkers)
    k = min(5000, len(loglike[:,0]))
    print "Using good samples from last %d iterations" %(k)
    match_cut = 1 - CR_99p9pc_threshold
    loglike_ok = loglike[-k:] # use samples from last k iterations
    matches_ok = np.sqrt(2*loglike_ok)/SNR
    sel = np.isfinite(loglike_ok) & (matches_ok > match_cut)
    chain_ok = np.array(map(lambda i: chain[:,-k:,i].T[sel], range(ndim))) # extract all 'good' samples
    idx = random.sample(xrange(len(chain_ok[0])), nwalkers) # get exactly nwalkers samples
    p0 = np.array(map(lambda i: chain_ok[:,i], idx))
  elif not auto_resume: raise RuntimeError("This cannot be!")
    

  #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4, args=[hps])
  if pt:
    ntemps = 5
    sampler = emcee.PTSampler(ntemps, nwalkers, ndim, logp=lnprior, logl=lnlikeMatch, loglargs=[hps], threads=nThreads)
    p0 = p0 * np.ones((ntemps, nwalkers, ndim))
  else:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobMatch, args=[hps], threads=nThreads)

  # Incrementally saving progress
  print "Saving MCMC chain to chain.dat as we go ..."
  f = open("chain.dat", "w")
  f.close()
  nout = 1000
  nout2 = 500
  
  if auto_resume: unique_id, outidx = chainid, partid
  else: 
    unique_id = int(1.0e7 * np.random.random())
    outidx= 0
  print " >> will write to output file %d-%d" % (unique_id, outidx)
  print "Dumping chain and logposterior to .npy files every %d iterations." %nout
  ii=1
  #for result in sampler.sample(p0, iterations=nsamples, storechain=False):
  for result in sampler.sample(p0, iterations=nsamples, storechain=True):
    position = result[0]
    ii+=1
    f = open("chain.dat", "a")
    chain_strings = []
    for k in range(position.shape[0]): # loop over number of walkers
      if (k == 0) and (ii % nout2 == 0):
        # Only works on OS X?
        # sys.stdout.write("\rIteration %d of %d" % ii, nsamples)
        # sys.stdout.flush()
        sys.stdout.write("Iteration %d of %d\n" %(ii, nsamples))
        sys.stdout.flush()
      #f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str, position[k]))))
      chain_strings.append("{0:4d} {1:s}\n".format(k, " ".join(map(str, position[k]))))
      # Dump samples in npy format every nout iterations
      if (k == 0) and (ii % nout == 0):
        np.save("chain%d-%d.npy" % (unique_id,outidx), sampler.chain[:,ii-nout:ii,:])
        np.save("loglike%d-%d.npy" % (unique_id,outidx), sampler.lnprobability.T[ii-nout:ii,:]) # it's really log posterior pdf
        outidx += 1
        print "** Saved chain.npy and loglike.npy. **"
      # Dump correlation length every nout iterations
      a_exp = np.nanmax(sampler.acor[0]) # we could take the max over all temperatures, but there may be nan's
      try:
        tmpf = open('autocorr.dat','a')
        tmpf.write('After burn-in, each chain produces one independent sample per {:g} steps\n'.format(a_exp))
        tmpf.close()
        if options.verbose:
          print('After burn-in, each chain produces one independent sample per {:g} steps'.format(a_exp))
      except: pass
    for chain_string in chain_strings: f.write(chain_string)
    f.close()

  if pt:
    print sampler.acor[0], max(sampler.acor[0])
    a_exp = max(sampler.acor[0]) # we could take the max over all temperatures, but there may be nan's
  else:
    a_exp = max(sampler.acor)
  # a_int = np.max([emcee.autocorr.integrated_time(sampler.chain[i]) for i in range(len(sampler.chain))], 0)
  print('A reasonable burn-in should be around {:d} steps'.format(int(10*a_exp)))
  try:
    tmpf = open('autocorr.dat','a')
    tmpf.write('After burn-in, each chain produces one independent sample per {:g} steps\n'.format(a_exp))
    tmpf.close()
    if options.verbose:
      print('After burn-in, each chain produces one independent sample per {:g} steps'.format(a_exp))
  except:
    pass
  # print [emcee.autocorr.integrated_time(sampler.chain[i], window=50) for i in range(len(sampler.chain))]
  # print [emcee.autocorr.integrated_time(sampler.chain[i], window=100) for i in range(len(sampler.chain))]
  # print [emcee.autocorr.integrated_time(sampler.chain[i], window=200) for i in range(len(sampler.chain))]

  # Save blobs
  np.save("chain.npy", sampler.chain)
  np.save("loglike.npy", sampler.lnprobability.T) # it's really log posterior pdf
  chain = sampler.chain
  loglike = sampler.lnprobability.T
  # normal shapes
  # chain = (100, 20, 4)
  # logl  = (20, 100)
  # pt shapes
  # chain = (5, 100, 20, 4)
  # logl  = (20, 100, 5)
  if pt: # get zero temperature chain
    chain = chain[0]
    loglike = loglike[:,:,0]
  print np.shape(chain)
  print np.shape(loglike)
else:
  print "Only running post-processing computation from ", post_process_directory
  print "Loading data ..."
  chain, loglike = read_run_part_names(post_process_directory, SNR, burnin=burnin)
  #chain = np.load(os.path.join(post_process_directory, "chain.npy"))
  #loglike = np.load(os.path.join(post_process_directory, "loglike.npy"))


# Start post-processing samples
print "Generating time plot ..."
pl.clf()
fig, axes = pl.subplots(5, 1, sharex=True, figsize=(8, 9))
axes[0].plot(chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(eta_true, color="#888888", lw=2)
axes[0].set_ylabel("$\eta$")
axes[0].set_ylim([0,0.25])

axes[1].plot(chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(Mc_true, color="#888888", lw=2)
axes[1].set_ylabel("$M_c$")

axes[2].plot(chain[:, :, 2].T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].axhline(chi1_true, color="#888888", lw=2)
axes[2].set_ylabel("$\chi_1$")
axes[2].set_ylim([-1,1])

axes[3].plot(chain[:, :, 3].T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].axhline(chi2_true, color="#888888", lw=2)
axes[3].set_ylabel("$\chi_2$")
axes[3].set_ylim([-1,1])

axes[4].plot(loglike, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].axhline(0.5*SNR**2, color="#888888", lw=2) # loglike value for match = 1
axes[4].set_ylabel("$\ln\Lambda = \\rho^2 m(s,h(\\theta)^2/2$")
axes[4].set_ylim([0.5*(SNR*0.9)**2,0.5*(SNR*1.01)**2]) # show from 0.9*SNR up to slightly beyond SNR for match=1
axes[4].set_xlabel("step number")

#fig.tight_layout(h_pad=0.0)
fig.savefig("match-time.png")
pl.close(fig)

# now use all good samples after burn-in
# be careful to combine the matches and samples correctly
samples_ok = chain[:, burnin:-1*(burnend+1), :]
loglike_ok = loglike[burnin:-1*(burnend+1),:] # this is log posterior pdf, but can get get back match modulo prior
matches_ok = np.sqrt(2*loglike_ok)/SNR
sel = np.isfinite(loglike_ok)  # could put something more stringent here! FIXME
#sel = np.isfinite(loglike_ok) & (matches_ok > 0.9)
print 'Keeping %d of %d samples' %(len(samples_ok[:,:,0].T[sel]), len(samples_ok[:,:,0].T.flatten()))
etaval = samples_ok[:,:,0].T[sel]
Mcval = samples_ok[:,:,1].T[sel]
chi1val = samples_ok[:,:,2].T[sel]
chi2val = samples_ok[:,:,3].T[sel]
loglikeval = loglike_ok[sel]
qval = qfun(etaval)
Mval = Mfun(Mcval,etaval)
m1val = m1fun(Mval,qval)
m2val = m2fun(Mval,qval)
chieffval = (m1val*chi1val + m2val*chi2val) / Mval
# FIXME
sel = np.isfinite(qval) & np.isfinite(Mval) & np.isfinite(m1val) & np.isfinite(m2val) & np.isfinite(chieffval)
qval, Mval, m1val, m2val, chieffval = qval[sel], Mval[sel], m1val[sel], m2val[sel], chieffval[sel]
etaval, Mcval, chi1val, chi2val = etaval[sel], Mcval[sel], chi1val[sel], chi2val[sel]
loglikeval = loglikeval[sel]

# if not post_process:
#   # Thin samples by auto-correlation time
#   etaval_thin  = etaval[::int(sampler.acor[0])]
#   Mcval_thin   = Mcval[::int(sampler.acor[1])]
#   chi1val_thin = chi1val[::int(sampler.acor[2])]
#   chi2val_thin = chi2val[::int(sampler.acor[3])]

m1_true = m1fun(M_true, q_true)
m2_true = m2fun(M_true, q_true)

print "Generating PDF plots ..."
plt.figure(int(np.random.random()*1e7))
plt.subplot(2,2,1)
plt.hist(Mcval, bins=50, alpha=0.6)
plt.gca().set_yscale('log')
plt.grid()
plt.xlabel('mchirp')

plt.subplot(2,2,2)
plt.hist(etaval, bins=50, alpha=0.6)
plt.gca().set_yscale('log')
plt.grid()
plt.xlabel('eta')

plt.subplot(2,2,3)
plt.hist(chi1val, bins=50, alpha=0.6)
plt.gca().set_yscale('log')
plt.grid()
plt.xlabel('chi1')

plt.subplot(2,2,4)
plt.hist(chi2val, bins=50, alpha=0.6)
plt.gca().set_yscale('log')
plt.grid()
plt.xlabel('chi2')
plt.savefig('PDFSamplingParameters.png')

plt.figure(int(np.random.random()*1e7))
plt.subplot(2,2,1)
plt.hist(m1val, bins=50, alpha=0.6)
plt.gca().set_yscale('log')
plt.grid()
plt.xlabel('m1')

plt.subplot(2,2,2)
plt.hist(m2val, bins=50, alpha=0.6)
plt.gca().set_yscale('log')
plt.grid()
plt.xlabel('m2')

plt.subplot(2,2,3)
plt.hist(qval, bins=50, alpha=0.6)
plt.gca().set_yscale('log')
plt.grid()
plt.xlabel('q')

plt.subplot(2,2,4)
plt.hist(chieffval, bins=50, alpha=0.6)
plt.gca().set_yscale('log')
plt.grid()
plt.xlabel('chi effective')
plt.savefig('PDFNonSampledParameters.png')

print "Generating triangle plots ..."
quantiles_68 = [0.16, 0.5, 0.84] # 1-sigma
quantiles_95 = [0.0228, 0.5, 0.9772] # 2-sigma ~ 95.4% : http://en.wikipedia.org/wiki/Percentile
quantiles=quantiles_95

samples_combined = np.array([qval, Mval, chi1val, chi2val]).T
print np.shape(samples_combined)
print samples_combined
try:
  fig = corner.corner(samples_combined, labels=["$q$", "$M$", "$\chi_1$", "$\chi_2$"], truths=[q_true, M_true, chi1_true, chi2_true], quantiles=quantiles, show_titles=True, title_args={"fontsize": 12}, verbose=True, plot_contours=True, plot_datapoints=True)
  fig.savefig("corner_q_M_chi1_chi2.png")
  pl.close(fig)
except: pass

samples_combined2 = np.array([etaval, Mcval, chi1val, chi2val]).T
fig = corner.corner(samples_combined2, labels=["$\eta$", "$\mathcal{M}$", "$\chi_1$", "$\chi_2$"], truths=[eta_true, Mc_true, chi1_true, chi2_true], range=[0.9999, 0.9999, 0.9999, 0.9999], quantiles=quantiles, show_titles=True, title_args={"fontsize": 12}, verbose=True, plot_contours=True, plot_datapoints=True)
fig.savefig("corner_eta_Mc_chi1_chi2.png")
pl.close(fig)

samples_combined3 = np.array([m1val, m2val, chi1val, chi2val]).T
fig = corner.corner(samples_combined3, labels=["$m_1$", "$m_2$", "$\chi_1$", "$\chi_2$"], truths=[m1_true, m2_true, chi1_true, chi2_true], quantiles=quantiles, show_titles=True, title_args={"fontsize": 12}, verbose=True, plot_contours=True, plot_datapoints=True)
fig.savefig("corner_m1_m2_chi1_chi2.png")
pl.close(fig)

print "Saving CR data ..."
np.savetxt('CR_data.dat', np.vstack([m1val, m2val, etaval, qval, Mval, Mcval, chi1val, chi2val, loglikeval]).T, delimiter='\t')


# Compute the quantiles.
print("Pretending that the distribution was normal, compute quantiles:\n")
eta_mcmc, Mc_mcmc, chi1_mcmc, chi2_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples_combined2, [16, 50, 84],axis=0)))
# http://en.wikipedia.org/wiki/Percentile normal distribution
# 50-16 = 34
# 84-50 = 34
print("""MCMC result 1 sigma:
    eta  = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    M_c  = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    chi1 = {4[0]} +{4[1]} -{4[2]} (truth: {5})
    chi2 = {6[0]} +{6[1]} -{6[2]} (truth: {7})
""".format(eta_mcmc, eta_true, Mc_mcmc, Mc_true, chi1_mcmc, chi1_true, chi2_mcmc, chi2_true))


eta_mcmc, Mc_mcmc, chi1_mcmc, chi2_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples_combined2, [2.28, 50, 97.72],axis=0)))

print("""MCMC result 2 sigma:
    eta  = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    M_c  = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    chi1 = {4[0]} +{4[1]} -{4[2]} (truth: {5})
    chi2 = {6[0]} +{6[1]} -{6[2]} (truth: {7})
""".format(eta_mcmc, eta_true, Mc_mcmc, Mc_true, chi1_mcmc, chi1_true, chi2_mcmc, chi2_true))

eta_mcmc, Mc_mcmc, chi1_mcmc, chi2_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples_combined2, [0.13, 50, 99.87],axis=0)))

print("""MCMC result 3 sigma:
    eta  = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    M_c  = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    chi1 = {4[0]} +{4[1]} -{4[2]} (truth: {5})
    chi2 = {6[0]} +{6[1]} -{6[2]} (truth: {7})
""".format(eta_mcmc, eta_true, Mc_mcmc, Mc_true, chi1_mcmc, chi1_true, chi2_mcmc, chi2_true))


def norm_dist_check(data, name):
  n = len(data)
  k = n/80 # how much to throw away to kill outliers very far from peak
  data = np.sort(data)[k:-k]
  mu, stdev = norm.fit(data)
  # Plot the histogram.
  f = pl.figure()
  pl.hist(data, bins=100, normed=True, alpha=0.6, color='b')
  # Plot the PDF.
  xmin, xmax = pl.xlim()
  x = np.linspace(xmin, xmax, 100)
  # improve to automatically discard samples that are in isolated regions separate from the peak region
  pfit = norm.pdf(x, mu, stdev)
  pl.figure()
  pl.plot(x, pfit, 'k', linewidth=2)
  title = "Fit results: mu = %.3f,  std = %.3f" % (mu, stdev)
  pl.title(title)
  pl.savefig("ROMCMC-match-norm-dist-check-"+name+".png")
  pl.close(f)
  return stdev

def stdev_drop_outliers(data, frac=80):
  # The smarter thing to do would be to trow away points with too low of a match
  n = len(data)
  k = n/frac # how much to throw away to kill outliers very far from peak
  data = np.sort(data)[k:-k]
  mu, stdev = norm.fit(data)
  return stdev

# data1D = [qval, etaval, Mval, Mcval, m1val, m2val, chi1val, chi2val, chieffval]
# names1D = ['q', 'eta', 'Mtot', 'Mchirp', 'm1', 'm2', 'chi1', 'chi2', 'chieff']
# stdevs = map(norm_dist_check, data1D, names1D)
#
# print 'Standard deviations:'
# print names1D
# print stdevs

def make_scatter_plot_for_match(etaS, McS, chi1S, chi2S, matchesS, filename):
  fig, axes = pl.subplots(1, 2, figsize=(15, 5))
  sc0=axes[0].scatter(etaS, McS, c=matchesS, cmap=pl.cm.spectral, edgecolors='None', alpha=0.75)
  axes[0].set_xlabel('$\eta$')
  axes[0].set_ylabel('$M_c$')
  fig.colorbar(sc0, ax=axes[0]);

  sc1=axes[1].scatter(chi1S, chi2S, c=matchesS, cmap=pl.cm.spectral, edgecolors='None', alpha=0.75)
  axes[1].set_xlabel('$\chi_1$')
  axes[1].set_ylabel('$\chi_2$')
  #axes[1].set_xlim([-1,1])
  #axes[1].set_ylim([-1,1])
  fig.colorbar(sc1, ax=axes[1]);

  fig.savefig(filename)
  pl.close(fig)

# Calculate lowest match and mask from safe lower bound on match
matches=np.sqrt(2*loglikeval)/SNR
print "Lowest match value found at", min(matches)
match_safe = 0.9
print "Keeping samples with match >", match_safe
mask = matches > match_safe

print "Generating scatter plot of match ..."
# FIXME
try:
  del chain, loglike, m1val, m2val, loglike_ok, samples_ok
  make_scatter_plot_for_match(etaval[mask], Mcval[mask], chi1val[mask], chi2val[mask], matches[mask], "scatter_match.png")
  make_scatter_plot_for_match(etaval, Mcval, chi1val, chi2val, matches, "scatter_match_all.png")
except: pass

print "All Done."
