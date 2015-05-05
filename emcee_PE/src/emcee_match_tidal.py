#!/usr/bin/env python

#
# MCMC python code using emcee to compute 4D credible regions
# This version uses parameters [eta, Mc, chi2=chiBH, Lambda] for tidal templates
#
# The idea is to use a match-based (zero-noise) likelihood 
# Lambda(s, theta) = exp(+(rho*match(s,h(theta)))^2/2)
# where rho is the desired snr.
#
# MP 11/2014 - 04/2015


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

parser = OptionParser()
parser.add_option("-n", "--nsamples", dest="nsamples", type="int",
                  help="How many samples to produce per walker", metavar="nsamples")
parser.add_option("-w", "--nwalkers", dest="nwalkers", type="int",
                  help="How many walkers to use.", metavar="nwalkers")
parser.add_option("-q", "--q_signal", dest="q_signal", type="float",
                  help="Asymmetric mass-ratio of signal (q>=1).", metavar="q_signal")
parser.add_option("-M", "--M_signal", dest="M_signal", type="float",
                  help="Total mass of signal in Msun.", metavar="M_signal")
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

# Additional options for tidal corection waveforms
parser.add_option("", "--inject-tidal", dest="inject_tidal", action="store_true",
                  help="Inject waveform LAL waveform with additional tidal corrections.", metavar="inject_tidal")
parser.add_option("-L", "--Lambda_signal", dest="Lambda_signal", type="float",
                  help="Tidal deformability Lambda of signal.", metavar="Lambda_signal")
parser.add_option("", "--Lambda_max", dest="Lambda_max", type="float",
                  help="Maximum tidal deformability Lambda of signal.", metavar="Lambda_max")
parser.add_option("-Y", "--Lambda_stdev_init", dest="Lambda_stdev_init", type="float",
                  help="Standard deviation for initial walker configurations in Lambda.", metavar="Lambda_stdev_init")

parser.set_defaults(nsamples=1000, nwalkers=100, q_signal=4.0, M_signal=100.0, 
  chi2_signal=0.0, f_min=10, f_max=2048, deltaF=0.5, 
  m1_min=6.0, m1_max=100.0, m2_min=6.0, m2_max=300.0, Mtot_max=500.0, SNR=15, burnin=250, eta_min=0.01,
  signal_approximant='lalsimulation.SEOBNRv2_ROM_DoubleSpin', 
  template_approximant='lalsimulation.SEOBNRv2_ROM_DoubleSpin',
  lalsim_psd='lalsimulation.SimNoisePSDaLIGOZeroDetHighPower',
  chi1_min=-1, chi1_max=0.99, chi2_min=-1, chi2_max=0.99,
  eta_stdev_init=0.15, Mc_stdev_init=5, chi2_stdev_init=0.4, post_process='',
  nThreads=1, pt=False,
  inject_tidal=False, Lambda_signal=500, Lambda_max=2500, Lambda_stdev_init=100)

(options, args) = parser.parse_args()

pt = options.pt
nThreads = options.nThreads # LAL calls fail with more than 1 thread. Why?

eta_stdev_init = options.eta_stdev_init
Mc_stdev_init = options.Mc_stdev_init
Lambda_stdev_init = options.Lambda_stdev_init
chi2_stdev_init = options.chi2_stdev_init

if Mc_stdev_init < 1e-6:
  print 'Mc_stdev_init < 1e-6!'
  print 'Setting Mc_stdev_init = 0.0001'
  Mc_stdev_init = 1e-4

if options.M_signal < options.m1_min + options.m2_min or options.M_signal > options.m1_max + options.m2_max:
  print 'Error: M_signal should be inside the mass prior [m1_min+m2_min, m1_max+m2_max]'
  sys.exit(-1)

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


# Parameter settings
nsamples = options.nsamples
nwalkers = options.nwalkers
burnin = options.burnin 

q_true = options.q_signal
M_true = options.M_signal
chi1_true = 0 # spin on NS = 0
chi2_true = options.chi2_signal
Lambda_true = options.Lambda_signal
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
s1z = 0 # spin on NS = 0
s2x = 0.
s2y = 0.
s2z = chi2_true
ampOrder = -1
phOrder = -1
distance = dist * 1.e6 * lal.PC_SI

# specify prior ranges
m1_min, m1_max = options.m1_min, options.m1_max
m2_min, m2_max = options.m2_min, options.m2_max
chi2_min, chi2_max = options.chi2_min, options.chi2_max
eta_min = options.eta_min
Lambda_max = options.Lambda_max


signal_approximant=eval(options.signal_approximant)
template_approximant=eval(options.template_approximant)

inject_tidal = options.inject_tidal
Lambda_signal = options.Lambda_signal


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
        [hps, hcs] = InjectTidalWaveform_ChooseFD(m1, m2, S1x=s1x, S1y=s1y, S1z=s1z, S2x=s2x, S2y=s2y, S2z=s2z, 
        Lambda=Lambda_signal, f_min=f_min, f_max=f_max, f_ref=f_ref, deltaF=deltaF, approximant=signal_approximant, make_plots=True)
      else:
        print 'Injecting FD approximant via ChooseFDWaveform()'
        Lambda_signal = 0
        print 'Setting Lambda = 0'
        [hps, hcs] = InjectWaveform_ChooseFD(m1, m2, S1x=s1x, S1y=s1y, S1z=s1z, S2x=s2x, S2y=s2y, S2z=s2z, 
        f_min=f_min, f_max=f_max, f_ref=f_ref, deltaF=deltaF, approximant=signal_approximant, make_plots=True)
    else: 
      # TD approximant : InjectWaveform will call SimInspiralTD()
      print 'Injecting TD approximant via SimInspiralTD()'
      Lambda_signal = 0
      print 'Setting Lambda = 0'
      [hps, hcs] = InjectWaveform(m1, m2, S1x=s1x, S1y=s1y, S1z=s1z, S2x=s2x, S2y=s2y, S2z=s2z, 
    f_min=f_min, f_max=f_max, f_ref=f_ref, deltaF=deltaF, approximant=signal_approximant, make_plots=True)    
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
  [hp, hc] = lalsimulation.SimInspiralChooseFDWaveform(phi0, deltaF, m1_SI, m2_SI, s1x, s1y, chi1, s2x, s2y, chi2, f_min, f_max, f_ref, distance, inclination, 0, 0, None, None, ampOrder, phOrder, template_approximant)
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
print "Signal snr: ", SNR
print "Using {0} as signal and {1} as templates.".format(options.signal_approximant, options.template_approximant)
print "Using PSD ", options.lalsim_psd
if inject_tidal:
  print "Lambda_signal = ", Lambda_signal

print 'Templates are LAL FD waveform with tidal corrections'
# Instantiate the class only once
tw = tidalWavs(approx=options.template_approximant.split('.')[1], verbose=False)

# Define the probability function as likelihood * prior.
ndim = 4
def lnprior(theta):
  [ eta, Mc, chi2, Lambda ] = theta
  if eta > 0.25 or eta < eta_min:
    return -np.inf
  q = qfun(eta)
  M = Mfun(Mc, eta)
  m1 = M*1.0/(1.0+q)
  m2 = M*q/(1.0+q)
  if m1 < m1_min or m1 > m1_max:
    return -np.inf
  if m2 < m2_min or m2 > m2_max:
    return -np.inf
  if M > options.Mtot_max:
    return -np.inf
  if chi2 < chi2_min or chi2 > chi2_max:
    return -np.inf
  
  # Additional priors to avoid calling tidal model outside of region of validity
  if eta < 6./49.:
    return -np.inf
  if chi2 > 0.75 or chi2 < -0.75:
    return -np.inf
  if Lambda < 0 or Lambda > Lambda_max:
    return -np.inf
  return 0.0

def lnlikeMatch(theta, s):
  # signal s
  [ eta, Mc, chi2, Lambda ] = theta
  if eta > 0.25 or eta < eta_min:
    return -np.inf
  q = qfun(eta)
  M = Mfun(Mc, eta)
  m1 = M*1.0/(1.0+q)
  m2 = M*q/(1.0+q)
  m1_SI = m1*lal.MSUN_SI
  m2_SI = m2*lal.MSUN_SI
  # print M, q, chi1, chi2
  # generate wf
  
  # LAL FD waveform with tidal corrections
  [hp, hc] = tw.getWaveform( M, eta, chi2, Lambda=Lambda, delta_f=deltaF, f_lower=f_min, f_final=f_max )
  hp = convert_FrequencySeries_to_lalREAL16FrequencySeries( hp ) # converts from pycbc.types.frequencyseries.FrequencySeries to COMPLEX16FrequencySeries
    
  psdfun = lalsim_psd
  ma = match_FS(s, hp, psdfun, zpf=2)
  if np.isnan(ma):
    print theta, ma
    print hp.data.data
  rho = SNR # use global variable for now
  return 0.5*(rho*ma)**2

def lnprobMatch(theta, s):
  lp = lnprior(theta)
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlikeMatch(theta, s)

if not post_process:
  p0 = emcee.utils.sample_ball(np.array([eta_true, Mc_true, chi2_true, Lambda_true]), np.array([eta_stdev_init, Mc_stdev_init, chi2_stdev_init, Lambda_stdev_init]), nwalkers)
  # may lead to warnings: ensemble.py:335: RuntimeWarning: invalid value encountered in subtract

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
  ii=0
  #for result in sampler.sample(p0, iterations=nsamples, storechain=False):
  for result in sampler.sample(p0, iterations=nsamples, storechain=True):
    position = result[0]
    ii+=1
    f = open("chain.dat", "a")
    for k in range(position.shape[0]): # loop over number of walkers
      if k == 0:
        # Only works on OS X?
        # sys.stdout.write("\rIteration %d of %d" % ii, nsamples)
        # sys.stdout.flush()
        sys.stdout.write("Iteration %d of %d\n" %(ii, nsamples))
        sys.stdout.flush()
      f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str, position[k]))))
    f.close()

  if pt:
    print sampler.acor[0], max(sampler.acor[0])
    a_exp = max(sampler.acor[0]) # we could take the max over all temperatures, but there may be nan's
  else:
    a_exp = max(sampler.acor)
  # a_int = np.max([emcee.autocorr.integrated_time(sampler.chain[i]) for i in range(len(sampler.chain))], 0)
  # print('A reasonable burn-in should be around {:d} steps'.format(int(10*a_exp)))
  try:
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
  chain = np.load(os.path.join(post_process_directory, "chain.npy"))
  loglike = np.load(os.path.join(post_process_directory, "loglike.npy"))


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
axes[2].axhline(chi2_true, color="#888888", lw=2)
axes[2].set_ylabel("$\chi_2$")
axes[2].set_ylim([-1,1])

axes[3].plot(chain[:, :, 3].T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].axhline(Lambda_true, color="#888888", lw=2)
axes[3].set_ylabel("$\Lambda$")
axes[3].set_ylim([-1,1])

axes[4].plot(loglike, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].axhline(0.5*SNR**2, color="#888888", lw=2) # loglike value for match = 1
axes[4].set_ylabel("$loglike = \\rho^2 m(s,h(\\theta)^2/2$")
axes[4].set_ylim([0.5*(SNR*0.9)**2,0.5*(SNR*1.01)**2]) # show from 0.9*SNR up to slightly beyond SNR for match=1
axes[4].set_xlabel("step number")

#fig.tight_layout(h_pad=0.0)
fig.savefig("match-time.png")
pl.close(fig)

# now use all good samples after burn-in
# be careful to combine the matches and samples correctly
samples_ok = chain[:, burnin:, :]
loglike_ok = loglike[burnin:,:] # this is log posterior pdf, but can get get back match modulo prior
matches_ok = np.sqrt(2*loglike_ok)/SNR
#sel = np.isfinite(loglike_ok)  # could put something more stringent here!
sel = np.isfinite(loglike_ok) & (matches_ok > 0.9)
print 'Keeping %d of %d samples' %(len(samples_ok[:,:,0].T[sel]), len(samples_ok[:,:,0].T.flatten()))
etaval = samples_ok[:,:,0].T[sel]
Mcval = samples_ok[:,:,1].T[sel]
chi2val = samples_ok[:,:,2].T[sel]
Lambdaval = samples_ok[:,:,3].T[sel]
loglikeval = loglike_ok[sel]
qval = qfun(etaval)
Mval = Mfun(Mcval,etaval)
m1val = m1fun(Mval,qval)
m2val = m2fun(Mval,qval)
chieffval = (m1val*0 + m2val*chi2val) / Mval # NS spin = 0

if not post_process:
  # Thin samples by auto-correlation time
  etaval_thin  = etaval[::int(sampler.acor[0])]
  Mcval_thin   = Mcval[::int(sampler.acor[1])]
  chi2val_thin = chi2val[::int(sampler.acor[2])]
  Lambdaval_thin = Lambdaval[::int(sampler.acor[3])]

m1_true = m1fun(M_true, q_true)
m2_true = m2fun(M_true, q_true)

print "Generating triangle plots ..."
quantiles_68 = [0.16, 0.5, 0.84] # 1-sigma
quantiles_95 = [0.0228, 0.5, 0.9772] # 2-sigma ~ 95.4% : http://en.wikipedia.org/wiki/Percentile
quantiles=quantiles_95

samples_combined = np.array([qval, Mval, chi2val, Lambdaval]).T
print np.shape(samples_combined)
print samples_combined
fig = triangle.corner(samples_combined, labels=["$q$", "$M$", "$\chi_2$", "$\Lambda$"], truths=[q_true, M_true, chi2_true, Lambda_true], quantiles=quantiles, show_titles=True, title_args={"fontsize": 12}, verbose=True, plot_contours=True, plot_datapoints=True)
fig.savefig("corner_q_M_chi2_Lambda.png")
pl.close(fig)

samples_combined2 = np.array([etaval, Mcval, chi2val, Lambdaval]).T
fig = triangle.corner(samples_combined2, labels=["$\eta$", "$\mathcal{M}$", "$\chi_2$", "$\Lambda$"], truths=[eta_true, Mc_true, chi2_true, Lambda_true], quantiles=quantiles, show_titles=True, title_args={"fontsize": 12}, verbose=True, plot_contours=True, plot_datapoints=True)
fig.savefig("corner_eta_Mc_chi2_Lambda.png")
pl.close(fig)

samples_combined3 = np.array([m1val, m2val, chi2val, Lambdaval]).T
fig = triangle.corner(samples_combined3, labels=["$m_1$", "$m_2$", "$\chi_2$", "$\Lambda$"], truths=[m1_true, m2_true, chi2_true, Lambda_true], quantiles=quantiles, show_titles=True, title_args={"fontsize": 12}, verbose=True, plot_contours=True, plot_datapoints=True)
fig.savefig("corner_m1_m2_chi2_Lambda.png")
pl.close(fig)

print "Saving CR data ..."
np.savetxt('CR_data.dat', np.vstack([m1val, m2val, etaval, qval, Mval, Mcval, chi2val, Lambdaval, loglikeval]).T, delimiter='\t')


# Compute the quantiles.
print("Pretending that the distribution was normal, compute quantiles:\n")
eta_mcmc, Mc_mcmc, chi2_mcmc, Lambda_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples_combined2, [16, 50, 84],axis=0)))
# http://en.wikipedia.org/wiki/Percentile normal distribution
# 50-16 = 34
# 84-50 = 34
print("""MCMC result 1 sigma:
    eta    = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    M_c    = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    chi2   = {4[0]} +{4[1]} -{4[2]} (truth: {5})
    Lambda = {6[0]} +{6[1]} -{6[2]} (truth: {7})
""".format(eta_mcmc, eta_true, Mc_mcmc, Mc_true, chi2_mcmc, chi2_true, Lambda_mcmc, Lambda_true))


eta_mcmc, Mc_mcmc, chi1_mcmc, chi2_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples_combined2, [2.28, 50, 97.72],axis=0)))

print("""MCMC result 2 sigma:
    eta    = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    M_c    = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    chi2   = {4[0]} +{4[1]} -{4[2]} (truth: {5})
    Lambda = {6[0]} +{6[1]} -{6[2]} (truth: {7})
""".format(eta_mcmc, eta_true, Mc_mcmc, Mc_true, chi2_mcmc, chi2_true, Lambda_mcmc, Lambda_true))

eta_mcmc, Mc_mcmc, chi2_mcmc, Lambda_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples_combined2, [0.13, 50, 99.87],axis=0)))

print("""MCMC result 3 sigma:
    eta    = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    M_c    = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    chi2   = {4[0]} +{4[1]} -{4[2]} (truth: {5})
    Lambda = {6[0]} +{6[1]} -{6[2]} (truth: {7})
""".format(eta_mcmc, eta_true, Mc_mcmc, Mc_true, chi2_mcmc, chi2_true, Lambda_mcmc, Lambda_true))


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
match_safe = 0.97
print "Keeping samples with match >", match_safe, " for scatter plot."
mask = matches > match_safe

print "Generating scatter plot of match ..."
make_scatter_plot_for_match(etaval[mask], Mcval[mask], chi2val[mask], Lambdaval[mask], matches[mask], "scatter_match.png")
make_scatter_plot_for_match(etaval, Mcval, chi2val, Lambdaval, matches, "scatter_match_all.png")

print "All Done."
