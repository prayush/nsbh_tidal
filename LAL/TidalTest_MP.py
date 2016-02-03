#!/usr/bin/env python

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import lal
from pycbc.waveform import *
from pycbc.types import *
import pycbc.psd as ppsd
from pycbc.filter import match

class tidalWavs():
    #{{{
    def __init__(self, approx='IMRPhenomC', verbose=True):
        self.verbose = verbose
        self.approx = approx
        # Amplitude correction factors
        # Define constants as per Eq 33 of Lackey et al
        if 'PhenomC' in approx:
            self.b0, self.b1, self.b2 = -64.985, -2521.8, 555.17
            self.c0, self.c1, self.c2 = -8.8093, 30.533, 0.64960
        elif 'SEOBNR' in approx:
            self.b0, self.b1, self.b2 = -1424.2, 6423.4, 0.84203
            self.c0, self.c1, self.c2 = -9.7628, 33.939, 1.0971
        # Phase correction factors
        if 'PhenomC' in approx:
            self.g0, self.g1, self.g2, self.g3 =\
                    -1.9051, 15.564, -0.41109, 5.7044
        elif 'SEOBNR' in approx: 
            self.g0, self.g1, self.g2, self.g3 =\
                    -4.6339, 27.719, 10.268, -41.741
    #
    def mtot_eta_to_m1_m2(self, M, eta):
        SqrtOneMinus4Eta = np.sqrt(1.-4.*eta)
        m1 = M * (1. + SqrtOneMinus4Eta) / 2.
        m2 = M * (1. - SqrtOneMinus4Eta) / 2.
        return m1, m2
    # 
    def tidalCorrectionAmplitude(self, mf, eta, sBH, tidalLambda,\
                                mfA=0.01):
        if mf <= mfA: return 1
        # Impose cutoffs on mass-ratio, and BH spins
        if eta < 6./49.: 
            print eta, 6./49.
            raise IOError("Eta too small")
        if sBH > 0.75: raise IOError("BH spin too large")
        if sBH < -0.75: raise IOError("BH spin too small")
        # Generate the amplitude factor
        C = np.exp(self.b0 + self.b1*eta + self.b2 *sBH) \
            + tidalLambda * np.exp(self.c0 + self.c1*eta + self.c2*sBH)
        D = 3.
        B = C * (mf - mfA)**D
        #print 'tidalCorrectionAmplitude: mf-mfA, B, C, D, Lambda:', mf-mfA, B, C, D, tidalLambda
        return np.exp(-eta * tidalLambda * B)
    #
    def tidalCorrectionPhase(self, mf, eta, sBH, tidalLambda,\
                             mfP=0.02):
        # First compute the PN inspiral phasing correction
        # see Eq. 7,8 of Lackey et al
        eta2, eta3 = eta**2, eta**3
        SqrtOneMinus4Eta = np.sqrt(1.-4.*eta)
        a0 = -12 * tidalLambda * ((1 + 7.*eta - 31 * eta2) \
                - SqrtOneMinus4Eta*(1 + 9. * eta - 11 * eta2))
        a1 = -(585. * tidalLambda / 28.) \
            * ((1. + 3775.*eta/234. - 389. * eta2 / 6. + 1376. * eta3 / 117.) \
            - SqrtOneMinus4Eta*(1 + 4243.*eta/234. - 6217 * eta2 / 234. - 10. * eta3/9.))
        pimf3rd = (np.pi * mf) ** (1./3.)
        psiT = 3.*(a0 * pimf3rd**5 + a1 * pimf3rd**7)/(128.*eta)
        # 
        if mf <= mfP: return psiT
        psi = psiT
        # Now compute the term propto derivative of psiT
        DpsiT = np.pi * (5.*a0 * pimf3rd**2 + 7.*a1 * pimf3rd**4)/(128.*eta)
        psi += ((mf - mfP) * DpsiT)
        # Now compute the phenomenological term
        G = np.exp(self.g0 + self.g1*eta + self.g2*sBH +\
                   self.g3*eta*sBH)
        H = 5./3.
        E = G * (mf - mfP)**H
        psi -= (eta * tidalLambda * E)
        #print 'tidalCorrectionPhase: a0, a1, G:', a0, a1, G
        #print 'tidalCorrectionPhase: g0, g1, g2, g3, eta, sBH', self.g0, self.g1, self.g2, self.g3, eta, sBH
        #print 'mfP, E', mfP, E
        #print 'psiT, DpsiT', psiT, DpsiT
        return psi
    #
    def getWaveform(self, M, eta, sBH, Lambda, f_lower=15.,\
                    delta_t=1./8192., delta_f=1./256, tidal=True):
        if self.approx in fd_approximants():
            m1, m2 = self.mtot_eta_to_m1_m2(M, eta)
            hp, hc = get_fd_waveform(approximant=self.approx,\
                        mass1=m1, mass2=m2, spin1z=sBH, spin2z=0,\
                        f_lower=f_lower, delta_f=delta_f)
        else: raise IOError("Approx not supported")
        if not tidal:
            if self.verbose:
                print "Returning WITHOUT tidal corrections"
                tid = int(0.1/M/lal.MTSUN_SI / delta_f)
                print hc[tid], hp[tid]
            return hp, hc
        # Tidal corrections to be incorporated
        freqs =  M * lal.MTSUN_SI * hp.sample_frequencies.data
        hpd, hcd = hp.data, hc.data
        ampC = np.array([self.tidalCorrectionAmplitude(mf, eta, sBH, Lambda) for mf in freqs])
        phsC = np.array([self.tidalCorrectionPhase(mf, eta, sBH, Lambda) for mf in freqs])
        Corr = np.array([np.complex(np.cos(phsC[ii]),-1*np.sin(phsC[ii])) for ii in range(len(phsC))])
        Corr = Corr * ampC
        hp = FrequencySeries(hp * Corr, delta_f=delta_f, epoch=hp._epoch, dtype=hp.dtype, copy=True)
        hc = FrequencySeries(hc * Corr, delta_f=delta_f, epoch=hp._epoch, dtype=hp.dtype, copy=True)
        if self.verbose:
            tid = int(0.1/M/lal.MTSUN_SI / delta_f)
            print ampC[tid], phsC[tid], Corr[tid]
            print hc[tid], hp[tid]
        return hp, hc

    def getWaveformAmpPhi(self, M, eta, sBH, Lambda, f_lower=15.,\
                    delta_t=1./8192., delta_f=1./256, tidal=True):
        if self.approx in fd_approximants():
            m1, m2 = self.mtot_eta_to_m1_m2(M, eta)
            hp, hc = get_fd_waveform(approximant=self.approx,\
                        mass1=m1, mass2=m2, spin1z=sBH, spin2z=0,\
                        f_lower=f_lower, delta_f=delta_f)
        else: raise IOError("Approx not supported")
        if not tidal:
            if self.verbose:
                print "Returning WITHOUT tidal corrections"
                tid = int(0.1/M/lal.MTSUN_SI / delta_f)
                print hc[tid], hp[tid]
            return hp, hc
        # Tidal corrections to be incorporated
        freqs =  M * lal.MTSUN_SI * hp.sample_frequencies.data
        hpd, hcd = hp.data, hc.data
        ampC = np.array([self.tidalCorrectionAmplitude(mf, eta, sBH, Lambda) for mf in freqs])
        phsC = np.array([self.tidalCorrectionPhase(mf, eta, sBH, Lambda) for mf in freqs])
        Corr = np.array([np.complex(np.cos(phsC[ii]),-1*np.sin(phsC[ii])) for ii in range(len(phsC))])
        Corr = Corr * ampC
        return ampC, phsC
     #}}}

sample_rate = 4096
time_length = 256 * 8
delta_f = 1./time_length
delta_f = 0.05 # MP temp
N = sample_rate * time_length
f_lower = 15.

Mmin, Mmax = [1.35, 5*1.35]
etamin, etamax = 5./36., 2./9.
smin, smax = -0.5, 0.75
tLambda = 600.

#print fd_approximants()
tw = tidalWavs(approx='SEOBNRv2_ROM_DoubleSpin_HI')

# MP specify parameters directy
M = 4.0
et = 0.2
s1 = 0.5
tLambda = 600.

hp, hc = tw.getWaveform(M, et, s1, tLambda, f_lower=f_lower, delta_f=delta_f)
fout = open('Lackey_M%.2f_Et%.2f_S%.2f_L%.1f_f%.1f.dat' % (M, et, s1, tLambda, f_lower), 'w+')
fout.write('# 1 Frequency [Hz]\n# 2 Real[h22]\n# 3 Imag[h22]\n')
tf, tre, tim = hp.sample_frequencies.data, np.real(hp.data), np.imag(hp.data) # MP return real and imaginary part of plus polarization
for i in range(len(tf)):
  fout.write('%.12e\t%.12e\t%.12e\n' % (tf[i], tre[i], tim[i]))
fout.close()

amp, phi = tw.getWaveformAmpPhi(M, et, s1, tLambda, f_lower=f_lower, delta_f=delta_f)
fout = open('Lackey_amp_phi_M%.2f_Et%.2f_S%.2f_L%.1f_f%.1f.dat' % (M, et, s1, tLambda, f_lower), 'w+')
fout.write('# 1 Frequency [Hz]\n# 2 A22\n# 3 phi22\n')
tf = hp.sample_frequencies.data
for i in range(len(tf)):
  fout.write('%.12e\t%.12e\t%.12e\n' % (tf[i], amp[i], phi[i]))
fout.close()



#psd=ppsd.from_txt('/home/prayush/research/advLIGO_PSDs/ZERO_DET_high_P.txt',\
#                          N/2+1, delta_f, low_freq_cutoff=f_lower-4.)

def random_match(outfile='match.dat'):
  #{{{
  rnd = np.random.random()
  M = rnd * (Mmax - Mmin) + Mmin
  et = rnd * (etamax - etamin) + etamin
  s1 = rnd * (smax - smin) + smin
  #
  hp, hc = tw.getWaveform(M, et, s1, tLambda, f_lower=f_lower)
  fout = open('Lackey_M%.2f_Et%.2f_S%.2f_L%.1f_f%.1f.dat' % (M, et, s1, tLambda, f_lower), 'w+')
  fout.write('# 1 Frequency [Hz]\n# 2 Real[h22]\n# 3 Imag[h22]\n')
  tf, tre, tim = hp.sample_frequencies.data, hp.data, hc.data
  for i in range(len(tf)):
    fout.write('%.12e\t%.12e\t%.12e\n' % (tf[i], tre[i], tim[i]))
  fout.close()

  tmp_hp = FrequencySeries( np.zeros(N/2+1), delta_f=delta_f, epoch=hp._epoch,\
              dtype=hp.dtype )
  tmp_hc = FrequencySeries( np.zeros(N/2+1), delta_f=delta_f, epoch=hp._epoch,\
              dtype=hp.dtype )
  tmp_hp[:len(hp)] = hp
  tmp_hc[:len(hc)] = hc
  hp, hc = tmp_hp, tmp_hc
  #
  hppp, hcpp = tw.getWaveform(M, et, s1, tLambda, tidal=False, f_lower=f_lower)
  tmp_hp = FrequencySeries( np.zeros(N/2+1), delta_f=delta_f, epoch=hp._epoch,\
              dtype=hp.dtype )
  tmp_hc = FrequencySeries( np.zeros(N/2+1), delta_f=delta_f, epoch=hp._epoch,\
              dtype=hp.dtype )
  tmp_hp[:len(hppp)] = hppp
  tmp_hc[:len(hcpp)] = hcpp
  hppp, hcpp = tmp_hp, tmp_hc
  #
  mm, _ = match(hp, hppp, psd=psd, low_frequency_cutoff=f_lower)
  #
  out = open(outfile,'a')
  out.write('%.12e\t%.12e\t%.12e\t%.12e\n' % (M, et, s1, mm))
  out.flush()
  out.close()
  #}}}

#############################################
#############################################
#############################################
#outfile = sys.argv[1]
#Nsims = int(sys.argv[2])

#for idx in range(Nsims):
#  random_match(outfile=outfile)

