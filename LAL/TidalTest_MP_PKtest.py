#!/usr/bin/env python

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import lal
import lalsimulation as LS
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
            # print eta, 6./49.
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
                # print hc[tid], hp[tid]
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

def matchMP(h1, h2, psdfun, deltaF, zpf=5):  
    """
    Compute the match between FD waveforms h1, h2
    :param h1, h2: data from frequency series [which start at f=0Hz]  :param psdfun: power spectral density as a function of frequency in Hz
    :param zpf:    zero-padding factor
    """
    assert(len(h1) == len(h2))
    n = len(h1)
    f = deltaF*np.arange(0,n)  
    psd_ratio = psdfun(100) / np.array(map(psdfun, f))
    psd_ratio[0] = psd_ratio[1] # get rid of psdfun(0) = nan
    h1abs = np.abs(h1)
    h2abs = np.abs(h2)
    norm1 = np.dot(h1abs, h1abs*psd_ratio)
    norm2 = np.dot(h2abs, h2abs*psd_ratio)
    integrand = h1 * h2.conj() * psd_ratio
    integrand_zp = np.concatenate([np.zeros(n*zpf), integrand, np.zeros(n*zpf)]) # zeropad it
    csnr = np.asarray(np.fft.fft(integrand_zp)) # complex snr
    return np.max(np.abs(csnr)) / np.sqrt(norm1*norm2)


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
tw = tidalWavs(approx='SEOBNRv2_ROM_DoubleSpin_HI', verbose=False)

# MP specify parameters directy
q = 6.0
mNS = 1.35
mBH = q*mNS
M = mNS + mBH
#print 'M', M
et = mNS*mBH / M**2
#print 'eta', et
s1 = 0.5
tLambda = 600.

def qfun(eta):
    return (1.0 + np.sqrt(1.0 - 4.0*eta) - 2.0*eta) / (2.0*eta)

def match_LAL_Python(M, et, s1, tLambda, f_lower=15, delta_f=0.5, save=False, zpf=2):
  #print M, et, s1, tLambda
  print '*'
  # produce Lackey waveform from Python code
  hp, hc = tw.getWaveform(M, et, s1, tLambda, f_lower=f_lower, delta_f=delta_f)
  tf, tre, tim = hp.sample_frequencies.data, np.real(hp.data), np.imag(hp.data) # MP return real and imaginary part of plus polarization
  hPy = tre + 1j*tim
  if save:
    fout = open('Lackey_M%.2f_Et%.2f_S%.2f_L%.1f_f%.1f.dat' % (M, et, s1, tLambda, f_lower), 'w+')
    fout.write('# 1 Frequency [Hz]\n# 2 Real[h22]\n# 3 Imag[h22]\n')
    for i in range(len(tf)):
      fout.write('%.12e\t%.12e\t%.12e\n' % (tf[i], tre[i], tim[i]))
    fout.close()

  amp, phi = tw.getWaveformAmpPhi(M, et, s1, tLambda, f_lower=f_lower, delta_f=delta_f)
  if save:
    fout = open('Lackey_amp_phi_M%.2f_Et%.2f_S%.2f_L%.1f_f%.1f.dat' % (M, et, s1, tLambda, f_lower), 'w+')
    fout.write('# 1 Frequency [Hz]\n# 2 A22\n# 3 phi22\n')
    tf = hp.sample_frequencies.data
    for i in range(len(tf)):
      fout.write('%.12e\t%.12e\t%.12e\n' % (tf[i], amp[i], phi[i]))
    fout.close()

  # produce Lackey waveform from LAL code and compare
  distance = 10e6*lal.PC_SI/5.
  inclination = 0
  q = qfun(et)
  mNS = M/(1+q)
  mBH = M*q/(1+q)
  Hp, Hc = LS.SimIMRLackeyTidal2013(0, delta_f, f_lower, tf[-1], f_lower, distance, inclination, mBH*lal.MSUN_SI, mNS*lal.MSUN_SI, s1, tLambda)
  hLAL = Hp.data.data + 1j*Hc.data.data
  if save:
    fout = open('Lackey_LAL_M%.2f_Et%.2f_S%.2f_L%.1f_f%.1f.dat' % (M, et, s1, tLambda, f_lower), 'w+')
    fout.write('# 1 Frequency [Hz]\n# 2 Real[h22]\n# 3 Imag[h22]\n')
    f = np.arange(Hp.data.length)*delta_f
    for i in range(len(f)):
      fout.write('%.12e\t%.12e\t%.12e\n' % (f[i], np.real(hLAL[i]), np.imag(hLAL[i])))
    fout.close()

  return matchMP(hPy, hLAL, LS.SimNoisePSDaLIGOZeroDetHighPower, delta_f, zpf=zpf)

def mismatch_LAL_Python(pars):
    [M, et, s1, tLambda] = pars
    f_lower=15
    delta_f=0.5
    save=False
    zpf=2
    return 1 - match_LAL_Python(M, et, s1, tLambda, f_lower=f_lower, delta_f=delta_f, save=save, zpf=zpf)

#############################################
#############################################
#############################################
#outfile = sys.argv[1]
#Nsims = int(sys.argv[2])

#for idx in range(Nsims):
#  random_match(outfile=outfile)

#print "mismatch:", 1.0 - match_LAL_Python(M, et, s1, tLambda, f_lower=f_lower, delta_f=delta_f, save=False, zpf=2)
#print "mismatch:", 1.0 - match_LAL_Python(M, et, s1, tLambda, f_lower=f_lower, delta_f=0.5, save=False, zpf=2)

# MPtest
# n = 2000
# q = np.random.uniform(1.0, 6.0, n)
# mNS = 1.35
# mBH = q*mNS
# M = mNS + mBH
# eta = mNS*mBH / M**2
# sBH = np.random.uniform(0.0, 0.7, n)
# tLambda = np.random.uniform(0.0, 3000, n)
#
#
# from multiprocessing import Pool
# p = Pool(4)
# data = p.map(mismatch_LAL_Python, ((M[i],eta[i],sBH[i],tLambda[i]) for i in np.arange(len(M))))
# print data
# np.savetxt('matches.dat', np.array([M, eta, sBH, tLambda, data]).T)
#
# # check case with ~ 0.008% mismatch
# M = 9.2314523641097139
# eta = 0.12485328039837687
# sBH = 0.1693455478691655
# Lambda = 1035.561578483296
# print match_LAL_Python(M, eta, sBH, Lambda, f_lower=15, delta_f=0.5, save=False, zpf=5)
# print match_LAL_Python(M, eta, sBH, Lambda, f_lower=15, delta_f=0.1, save=True, zpf=5)
#


###########################################################
import numpy as np
from pycbc.waveform import *
import pycbc.psd
from pycbc.types import *
from pylab import *

import sys
# sys.path.append('/home/prayush/research/NSBH/TidalParameterEstimation/nsbh_tidal/emcee_PE/src/')
# sys.path.append('/home/prayush/research/NSBH/TidalParameterEstimation/nsbh_tidal/emcee_PE/scripts//')

# from utils import *
# from match import *
# from injection import *

### FILTERING PARAMETERS
filter_waveform_length = 512
filter_sample_rate = 8192*4
filter_N = filter_waveform_length * filter_sample_rate
filter_n = int(filter_N/2) + 1
df   = 1./filter_waveform_length
f_min = 14.0
f_max = 4096.0

### GET PSD
psd = pycbc.psd.from_string('aLIGOZeroDetHighPower', filter_n, 1./filter_waveform_length, f_min)
psd *= DYN_RANGE_FAC**2
psd = FrequencySeries(psd,delta_f=psd.delta_f,dtype=complex128)

tw = tidalWavs(approx='SEOBNRv2_ROM_DoubleSpin_HI', verbose=False)

### PHYSICAL PARAMETERS
m1   = 2*1.35
m2   = 1.35
M    = m1 + m2
q    = m1 / m2
eta  = q / (1. + q)**2
s1z  = 0.5
Lambda = 1500

q_range = np.arange(2, 6.5, 0.5)

fout = open('matches.dat','w')
matches = []
for qq in q_range:
  print "for mass-ratio = %f" % qq
  hpL, _ = get_fd_waveform(approximant='Lackey_Tidal_2013_SEOBNRv2_ROM',\
                  mass1=m2*qq, mass2=m2, spin1z=s1z, lambda2=Lambda, f_lower=f_min, delta_f=df)
  hpP, _ = tw.getWaveform(m2*(1.+qq), qq/(1.+qq)**2, s1z, Lambda=Lambda, delta_f=df, f_lower=f_min) #, f_final=f_max
  hpP2 = FrequencySeries(zeros(len(hpL)), delta_f=hpP.delta_f, dtype=complex_same_precision_as(hpP))
  hpP2[:len(hpP)] = hpP
  m, _ = match(hpL, hpP2, psd=psd, low_frequency_cutoff=f_min+1.)
  fout.write('%.12e\t%.12e\n' % (qq, m))
  fout.flush()
  matches.append(m)
  print " .. match = ", m
      
fout.close()

mismatchQvary = [1. - x for x in matches]

semilogy(q_range[:len(mismatchQvary)], mismatchQvary, 'k-o')
xlabel('MASS-RATIO =m1/m2 (m2=1.35 fixed)')
ylabel('1 - OVERLAP')
grid()
###########################################################