#!/usr/bin/python

# MP: script to set up a series of runs in total mass for fixed q, chi1, chi2 and SNR
#
# Start the runs like so:
# for d in dirname/*/ ; do echo $d; cd $d && bash ./run.sh && cd -; done

import os

chi1 = 0.9 # small BH
chi2 = 0.9 # large BH
q = 2
mvec=[6, 12, 20, 50]
SNR=30

eta = q/(1.0 + q)**2

EXE="/home/spxmp/spin_estimation_emcee/emcee_match_aligned.py"


for m in mvec:
  path = 'q=%f_chi1=%f_chi2=%f_SNR%d_DS/M%dMsun' %(q,chi1,chi2,SNR,m)
  filename = 'run.sh'

  Mc = m * eta ** 3.0/5.0

  if not os.path.exists(path):
    os.makedirs(path)

  script = '''\
nohup %s \
  --signal_approximant lalsimulation.SEOBNRv2_ROM_DoubleSpin_LM \
  --template_approximant lalsimulation.SEOBNRv2_ROM_DoubleSpin_LM \
  --psd lalsimulation.SimNoisePSDaLIGOZeroDetHighPower \
  --q_signal %f \
  --M_signal %f \
  --snr %f \
  --chi1_signal %f \
  --chi2_signal %f \
  --f_min 10.0 \
  --f_max 4096.0 \
  --deltaF 0.5 \
  --m1_min 1.0 \
  --m1_max %f \
  --m2_min 1.0 \
  --m2_max %f \
  --nwalkers 100  \
  --nsamples 3000 \
  --burnin 500 \
  --chi1_min -0.9999 \
  --chi1_max 0.98999 \
  --chi2_min -0.9999 \
  --chi2_max 0.98999 \
  --Mc_stdev_init %f \
> out 2>&1 &
  '''
  buff = script %(EXE, q, m, SNR, chi1, chi2, 4*m, 4*m, Mc**2 / 50.0)
  with open(os.path.join(path, filename), 'wb') as temp_file:
    temp_file.write(buff)



