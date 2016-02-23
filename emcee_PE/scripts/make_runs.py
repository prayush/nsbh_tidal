#!/usr/bin/python

# MP: script to set up a series of runs in total mass for fixed q, chi1, chi2 and SNR
#
# Start the runs like so:
# for d in dirname/*/ ; do echo $d; cd $d && bash ./run.sh && cd -; done

import os, sys
import commands as cmd
import numpy as np


def tidal_submit_string( idir, exe,\
                  Lambda, q, m, SNR, chi1, chi2,\
                  m1max, m2max, Mcstdev, inject_tidal=True,\
                  chi1max=0.98, chi1min=-0.99, chi2max=0.98, chi2min=-0.99,\
                  chi2only=False,\
                  LambdaMax=2000, Lambdastdev=100, recover_tidal=True,\
                  Nwalkers=100, Nsamples=3000, Nburnin=500,\
                  postprocess_only=False ):
    #{{{
    if postprocess_only: print "ONLY POST-Processing!!"
    print Mcstdev
    script = '''\
universe=vanilla
getenv=True
initialdir=%s
\n
executable=%s
arguments= --signal_approximant lalsimulation.SEOBNRv2_ROM_DoubleSpin_LM \
--template_approximant lalsimulation.SEOBNRv2_ROM_DoubleSpin_LM \
--psd lalsimulation.SimNoisePSDaLIGOZeroDetHighPower \
--q_signal %f \
--M_signal %f \
--snr %f \
'''
    if not recover_tidal:
      script += '''\
--chi1_signal %f \
'''
    script += '''\
--chi2_signal %f \
--f_min 15.0 \
--f_max 4096.0 \
--deltaF 0.5 \
--m1_min 1.2 \
--m1_max %f \
--m2_min 1.2 \
--m2_max %f \
--nwalkers %d  \
--nsamples %d \
--burnin %d \
--chi1_min %f \
--chi1_max %f \
--chi2_min %f \
--chi2_max %f \
--Mc_stdev_init %f \
'''
    #
    if inject_tidal:
      script += '''\
--inject-tidal \
--Lambda_signal %f \
'''
    #
    if recover_tidal:
      script += '''\
--Lambda_max %f \
--Lambda_stdev_init %f \
'''
    #
    if postprocess_only:
      script += '''\
-R . \
'''
    #
    script += '''\
\n
log=logs/job.log
output=logs/job.out
error=logs/job.err
notification=Always
notify_user=prkumar@cita.utoronto.ca
queue 1
  '''
    #
    if inject_tidal and not recover_tidal:
      buff = script %( idir, exe, q, m, SNR, chi1, chi2, m1max, m2max,\
        Nwalkers, Nsamples, Nburnin, chi1min, chi1max, chi2min, chi2max, Mcstdev, Lambda)
    elif inject_tidal and recover_tidal:
      buff = script %( idir, exe, q, m, SNR, chi2, m1max, m2max,\
        Nwalkers, Nsamples, Nburnin, chi1min, chi1max, chi2min, chi2max, Mcstdev, Lambda, LambdaMax, Lambdastdev )
    elif recover_tidal:
      buff = script %( idir, exe, q, m, SNR, chi2, m1max, m2max,\
        Nwalkers, Nsamples, Nburnin, chi1min, chi1max, chi2min, chi2max, Mcstdev, LambdaMax, Lambdastdev)
    else:
      buff = script %( idir, exe, q, m, SNR, chi1, chi2, m1max, m2max,\
        Nwalkers, Nsamples, Nburnin, chi1min, chi1max, chi2min, chi2max, Mcstdev)
    return buff
    #}}}


def submit_string( idir, exe,\
                  Lambda, q, m, SNR, chi1, chi2,\
                  m1max, m2max, Mcstdev, inject_tidal=True,\
                  chi1max=0.98, chi1min=-0.99, chi2max=0.98, chi2min=-0.99,\
                  chi2only=False,\
                  LambdaMax=2000, Lambdastdev=100, recover_tidal=True,\
                  Nwalkers=100, Nsamples=3000, Nburnin=500,\
                  postprocess_only=False ):
    #{{{
    if postprocess_only: print "ONLY POST-Processing!!"
    print Mcstdev
    script = '''\
universe=vanilla
getenv=True
initialdir=%s
\n
executable=%s
arguments= --signal_approximant lalsimulation.SEOBNRv2_ROM_DoubleSpin_LM \
--template_approximant lalsimulation.SEOBNRv2_ROM_DoubleSpin_LM \
--psd lalsimulation.SimNoisePSDaLIGOZeroDetHighPower \
--q_signal %f \
--M_signal %f \
--snr %f \
--chi1_signal %f \
--chi2_signal %f \
--f_min 15.0 \
--f_max 4096.0 \
--deltaF 0.5 \
--m1_min 1.2 \
--m1_max %f \
--m2_min 1.2 \
--m2_max %f \
--nwalkers %d  \
--nsamples %d \
--burnin %d \
--chi1_min %f \
--chi1_max %f \
--chi2_min %f \
--chi2_max %f \
--Mc_stdev_init %f \
'''
    #
    if chi2only:
        script += '''\
--chi2_only \
'''
    if inject_tidal:
        script += '''\
--inject-tidal \
--Lambda_signal %f \
'''
    #
    if postprocess_only:
        script += '''\
-R . \
'''
    #
    script += '''\
\n
log=logs/job.log
output=logs/job.out
error=logs/job.err
notification=Always
notify_user=prkumar@cita.utoronto.ca
queue 1
  '''
    #
    if inject_tidal:
      buff = script %( idir, exe, q, m, SNR, chi1, chi2, m1max, m2max,\
        Nwalkers, Nsamples, Nburnin, chi1min, chi1max, chi2min, chi2max, Mcstdev, Lambda)
    else:
      buff = script %( idir, exe, q, m, SNR, chi1, chi2, m1max, m2max,\
        Nwalkers, Nsamples, Nburnin, chi1min, chi1max, chi2min, chi2max, Mcstdev)
    return buff
    #}}}

def get_simdirname(q, mNS, chi2, Lambda, SNR, Nw, Ns):
  #{{{
  return 'q%.2f_mNS%.2f_chiBH%.2f_Lambda%.1f_SNR%.1f/NW%d_NS%d'\
              % (q, mNS, chi2, Lambda, SNR, Nw, Ns)
  #}}}


######################################################
# Set up parameters of signal
######################################################
chi1 = 0.   # small BH
chi2vec = [-0.5, 0, 0.5, 0.74999]  # larger BH
mNS = 1.35
qvec = [2, 3, 4, 5]
Lambdavec = [500, 800, 1000, 1500, 2000]
SNRvec = [20, 30, 50, 70, 90, 120]

inject_tidal = True

if not inject_tidal: Lambdavec = [0]
######################################################
# Set up parameters of templates
######################################################
# Taking an uninformed prior
m1min = 1.2
m1max = 15.
m2min = 1.2 
m2max = 25

chi1min = -0.99
chi1max = 0.98
chi2min = -0.99
chi2max = 0.98
chi2only = True

LambdaMax = 4000*2
Lambdastdev = 100

recover_tidal = False

######################################################
# Set up RUN parameters
######################################################
if recover_tidal:
  EXE = "/home/prayush/src/nsbh_tidal/emcee_PE/src/emcee_match_tidal.py"
  subfunc = tidal_submit_string
else:
  EXE = "/home/prayush/src/nsbh_tidal/emcee_PE/src/emcee_match_aligned.py"
  subfunc = submit_string

if inject_tidal: sigstring = 'T'
else: sigstring = 'N'
if recover_tidal: tmpstring = 'T'
else: tmpstring = 'N'
simstring = sigstring + tmpstring + '_'

PWD = cmd.getoutput('pwd')
filename = 'run.sub'
Nwalkers = [100]
Nsamples = [2000]
Nburnin  = 500
postprocess_only = False

######################################################
# Set up RUNs
######################################################
f = open(simstring +\
      '_q%.1f_%.1f__chiBH%.2f_%.2f__Lambda%.1f_%.1f_SNR%.1f_%.1f__NW%d_%d__NS%d_%d_'\
      % (min(qvec), max(qvec), min(chi2vec), max(chi2vec),\
         min(Lambdavec), max(Lambdavec), min(SNRvec), max(SNRvec),\
         min(Nwalkers), max(Nwalkers), min(Nsamples), max(Nsamples))\
      + '_SEOBNRv2ROM.dag','w')

for q in qvec:
  for chi2 in chi2vec:
    for Lambda in Lambdavec:
      for SNR in SNRvec:
        for Ns in Nsamples:
          for Nw in Nwalkers:
            eta = q / (1. + q)**2
            m = (1. + q) * mNS
            Mc = m * eta**0.6
            Mcstdev = 0.15 * Mc
            path = simstring
            path += 'q%.2f_mNS%.2f_chiBH%.2f_Lambda%.1f_SNR%.1f/NW%d_NS%d' % (q, mNS, chi2, Lambda, SNR, Nw, Ns)
            if not os.path.exists(path): os.makedirs(path)
            if not os.path.exists(path+'/logs'): os.makedirs(path+'/logs')
            #
            buff = subfunc(PWD+'/'+path, EXE, \
                    Lambda, q, m, SNR, chi1, chi2,\
                    m1max, m2max, Mcstdev, inject_tidal=inject_tidal,\
                    LambdaMax=LambdaMax, Lambdastdev=Lambdastdev,\
                    recover_tidal=recover_tidal,\
                    chi1min=chi1min, chi1max=chi1max, chi2min=chi2min, chi2max=chi2max,\
                    chi2only=chi2only,\
                    Nwalkers=Nw, Nsamples=Ns, Nburnin=Nburnin,\
                    postprocess_only=postprocess_only)
            #
            with open(os.path.join(path, filename), 'wb') as temp_file:
              temp_file.write(buff)
            #
            JobName = 'q%.2f_mNS%.2f_chiBH%.2f_Lambda%.1f_SNR%.1f_NW%d_NS%d' % (q, mNS, chi2, Lambda, SNR, Nw, Ns)
            f.write('Job %s %s/%s\n' % (JobName, path, filename))
            f.write('Retry %s 1\n' % JobName)
            f.write('PRIORITY %s 1000\n\n' % JobName)

f.close()

