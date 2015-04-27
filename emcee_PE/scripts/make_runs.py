#!/usr/bin/python

# MP: script to set up a series of runs in total mass for fixed q, chi1, chi2 and SNR
#
# Start the runs like so:
# for d in dirname/*/ ; do echo $d; cd $d && bash ./run.sh && cd -; done

import os
import commands as cmd
import numpy as np

chi1 = 0.   # small BH
chi2 = 0.5  # larger BH
q = 3
mNS = 1.35
mvec=[(q+1)*mNS]
SNR=30

eta = q/(1.0 + q)**2

EXE="/home/prayush/src/nsbh_tidal/emcee_PE/src/emcee_match_aligned.py"
PWD = cmd.getoutput('pwd')

for m in mvec:
  path = 'q%.2f_chi%.2f_chi%.2f_SNR%.1f_DS/MBH%.2f_MNS%.2f' %\
                    (q,chi1,chi2,SNR, m - mNS, mNS)
  filename = 'run.sub'

  Mc = m * eta ** 3.0/5.0

  if not os.path.exists(path):
    os.makedirs(path)
  if not os.path.exists(path+'/logs'):
    os.makedirs(path+'/logs')

  script = '''\
universe=vanilla
getenv=True
initialdir=%s
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
--nwalkers 100  \
--nsamples 3000 \
--burnin 500 \
--chi1_min -0.99 \
--chi1_max 0.98 \
--chi2_min -0.99 \
--chi2_max 0.98 \
--Mc_stdev_init %f

log=logs/job.log
output=logs/job.out
error=logs/job.err
notification=never
queue 1
  '''
  buff = script %(PWD+'/'+path, EXE, q, m, SNR, chi1, chi2, 4*m, 4*m, Mc**2 / 50.0)
  with open(os.path.join(path, filename), 'wb') as temp_file:
    temp_file.write(buff)


