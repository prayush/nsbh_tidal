#!/usr/bin/python

# MP: script to set up a series of runs in total mass for fixed q, chi1, chi2 and SNR
#
# Start the runs like so:
# for d in dirname/*/ ; do echo $d; cd $d && bash ./run.sh && cd -; done

import os, sys
import commands as cmd
import numpy as np

def submit_string( idir, exe,\
                  Lambda, q, m, SNR, chi1, chi2,\
                  m1max, m2max, Mcstdev, inject_tidal=True,\
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
--chi1_min -0.99 \
--chi1_max 0.98 \
--chi2_min -0.99 \
--chi2_max 0.98 \
--Mc_stdev_init %f \
'''
    #
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
notification=never
queue 1
  '''
    #
    if inject_tidal:
      buff = script %( idir, exe, q, m, SNR, chi1, chi2, m1max, m2max,\
        Nwalkers, Nsamples, Nburnin, Mcstdev, Lambda)
    else:
      buff = script %( idir, exe, q, m, SNR, chi1, chi2, m1max, m2max,\
        Nwalkers, Nsamples, Nburnin, Mcstdev)
    return buff
    #}}}


######################################################
# Set up parameters of signal
######################################################
chi1 = 0.   # small BH
chi2 = 0.5  # larger BH
mNS = 1.35
q = 3
qvec = [2,3,4,5]
Lambdavec = [200, 500, 1000]
SNRvec = [30, 100]
inject_tidal = False

######################################################
# Set up parameters of signal
######################################################
# Taking an uninformed prior
m1min = 1.2
m1max = 30.
m2min = 1.2 
m2max = 30

######################################################
# Set up RUN parameters
######################################################
EXE = "/home/prayush/src/nsbh_tidal/emcee_PE/src/emcee_match_aligned.py"
PWD = cmd.getoutput('pwd')
filename = 'run.sub'
Nwalkers = [100, 150, 200]
Nsamples = 3000
Nburnin  = 500
postprocess_only = False


for Nw in Nwalkers:
  for Lambda in Lambdavec:
    for SNR in SNRvec:
      eta = q / (1. + q)**2
      m = (1. + q) * mNS
      Mc = m * eta**0.6
      path = 'q%.2f_chi%.2f_chi%.2f_Lambda%.1f_SNR%.1f_DS/MBH%.2f_MNS%.2f_NW%d' %\
                    (q, chi1, chi2, Lambda, SNR, m - mNS, mNS, Nw)
      if not os.path.exists(path): os.makedirs(path)
      if not os.path.exists(path+'/logs'): os.makedirs(path+'/logs')
      #
      buff = submit_string(PWD+'/'+path, EXE, \
          Lambda, q, m, SNR, chi1, chi2, m1max, m2max, 0.2 * Mc, \
          inject_tidal=inject_tidal,\
          Nwalkers=Nw, Nsamples=Nsamples, Nburnin=Nburnin)
      #
      with open(os.path.join(path, filename), 'wb') as temp_file:
        temp_file.write(buff)
  #break


