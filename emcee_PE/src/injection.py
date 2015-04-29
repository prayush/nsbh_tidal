# MP 11/2014 - 04/2015

from __future__ import division

from matplotlib import use, rc
use('Agg')
rc('text', usetex=True)
from matplotlib import pyplot as plt

import numpy as np
from scipy import interpolate
import lal
from lal import MSUN_SI, PC_SI
import lalsimulation as lalsim

from match import *

##############################################################
# Additional imports for Tidal waveform generation
from pycbc.waveform import *
from TidalWaveforms import *

def convert_FrequencySeries_to_lalREAL16FrequencySeries( h, name=None ):
  tmp = lal.CreateCOMPLEX16Sequence( len(h) )
  tmp.data = np.array(h.data)
  hnew = lal.COMPLEX16FrequencySeries()
  hnew.data = tmp
  hnew.deltaF = h.delta_f
  hnew.epoch = h._epoch
  if name is not None: hnew.name = name
  return hnew

#############################################################
#############################################################

def InjectWaveform(m1, m2, S1x=0, S1y=0, S1z=0, S2x=0, S2y=0, S2z=0, f_min=10.0, f_max=2048.0, f_ref=0.0, deltaF=0.1, approximant=lalsim.SEOBNRv2, make_plots=False):
  fs = 2*f_max
  deltaT = 1./fs
  phiRef = 0.0
  m1SI = m1 * MSUN_SI
  m2SI = m2 * MSUN_SI
  r = 1e6 * PC_SI
  z = 0.0
  i = 1.0
  lambda1 = 0.0
  lambda2 = 0.0
  waveFlags = None
  nonGRparams = None
  amplitudeO = -1
  phaseO = -1

  # produce FD or TD waveform
  Hp, Hc = lalsim.SimInspiralFD(phiRef, deltaT, m1SI, m2SI, S1x, S1y, S1z, S2x, S2y, S2z, 
                       f_min*0.8, f_ref, r, z, i, lambda1, lambda2, waveFlags, nonGRparams, amplitudeO, phaseO, approximant)

  f = np.arange(Hp.data.length) * Hp.deltaF

  # Split into amplitude and phase and interpolate
  # h+
  amp_hp = np.abs(Hp.data.data)
  phi_hp = np.unwrap(np.angle(Hp.data.data))
  amp_hpI = interpolate.interp1d(f, amp_hp)
  phi_hpI = interpolate.interp1d(f, phi_hp)
  # hc
  amp_hc = np.abs(Hc.data.data)
  phi_hc = np.unwrap(np.angle(Hc.data.data))
  amp_hcI = interpolate.interp1d(f, amp_hc)
  phi_hcI = interpolate.interp1d(f, phi_hc)

  # Resample to desired deltaF
  n = int(1 + (f_max - 0) / deltaF)
  f_in = np.arange(n) * deltaF
  mask = (f_in < f_min) | (f_in > f_max)
  # h+
  amp_hp_out = amp_hpI(f_in)
  phi_hp_out = phi_hpI(f_in)
  h_hp_out = amp_hp_out * np.exp(1j*phi_hp_out)
  h_hp_out[mask] = 0.0
  # hc
  amp_hc_out = amp_hcI(f_in)
  phi_hc_out = phi_hcI(f_in)
  h_hc_out = amp_hc_out * np.exp(1j*phi_hc_out)
  h_hc_out[mask] = 0.0
  
  # Create new frequency series & resize it to follow lalsimulation convention
  n_pow2 = int(1 + 2**np.ceil(np.log2(n-1))) # make length 1 + power of two
  # h+
  H_hp_out = lal.CreateCOMPLEX16FrequencySeries('Resampled Hp strain', lal.LIGOTimeGPS(0,0), 0.0, deltaF, lal.StrainUnit, n)
  H_hp_out.epoch = Hp.epoch
  H_hp_out.sampleUnits = Hp.sampleUnits
  H_hp_out.data.data = h_hp_out
  H_hp_out2 = lal.ResizeCOMPLEX16FrequencySeries(H_hp_out, 0, n_pow2)
  # hc
  H_hc_out = lal.CreateCOMPLEX16FrequencySeries('Resampled Hc strain', lal.LIGOTimeGPS(0,0), 0.0, deltaF, lal.StrainUnit, n)
  H_hc_out.epoch = Hc.epoch
  H_hc_out.sampleUnits = Hc.sampleUnits
  H_hc_out.data.data = h_hc_out
  H_hc_out2 = lal.ResizeCOMPLEX16FrequencySeries(H_hc_out, 0, n_pow2)
  
  if make_plots:
    # Make plots of injection
    plt.loglog(f, amp_hp)
    plt.loglog(f_in, amp_hp_out, 'r--')
    plt.xlabel(r'$f[Hz]$')
    plt.ylabel(r'$|\tilde h|$')
    plt.xlim([f_min, f_max])
    plt.savefig('injection_amplitude.png')
    plt.clf()

    plt.semilogx(f, phi_hp)
    plt.semilogx(f_in, phi_hp_out, 'r-.')
    plt.xlabel(r'$f[Hz]$')
    plt.ylabel(r'$\phi[\tilde h]$')
    plt.xlim([f_min, f_max])
    plt.savefig('injection_phase.png')
    plt.clf()

    plt.plot(f, np.real(Hp.data.data))
    plt.plot(f_in, np.real(h_hp_out), 'r-.')
    plt.xlim([f_min, 3*f_min])
    plt.savefig('injection_re_h.png')
    plt.clf()
  
  return [H_hp_out2, H_hc_out2]

def InjectWaveform_ChooseFD(m1, m2, S1x=0, S1y=0, S1z=0, S2x=0, S2y=0, S2z=0, f_min=10.0, f_max=2048.0, f_ref=0.0, deltaF=0.1, approximant=lalsim.SEOBNRv2, distance=1e6*PC_SI, make_plots=False):
  phiRef = 0.0
  m1SI = m1 * MSUN_SI
  m2SI = m2 * MSUN_SI
  z = 0.0
  r = distance
  i = 1.0
  lambda1 = 0.0
  lambda2 = 0.0
  waveFlags = None
  nonGRparams = None
  amplitudeO = -1
  phaseO = -1
  [Hp, Hc] = lalsim.SimInspiralChooseFDWaveform(phiRef, deltaF, m1SI, m2SI, S1x, S1y, S1z, S2x, S2y, S2z, f_min, f_max, f_ref, r, i, 0, 0, None, None, amplitudeO, phaseO, approximant)

  if make_plots:
    f = np.arange(Hp.data.length) * Hp.deltaF
    amp_hp = np.abs(Hp.data.data)
    phi_hp = np.unwrap(np.angle(Hp.data.data))
    
    # Make plots of injection
    plt.loglog(f, amp_hp)
    plt.xlabel(r'$f[Hz]$')
    plt.ylabel(r'$|\tilde h|$')
    plt.xlim([f_min, f_max])
    plt.savefig('injection_amplitude.png')
    plt.clf()

    plt.semilogx(f, phi_hp)
    plt.xlabel(r'$f[Hz]$')
    plt.ylabel(r'$\phi[\tilde h]$')
    plt.xlim([f_min, f_max])
    plt.savefig('injection_phase.png')
    plt.clf()

    plt.plot(f, np.real(Hp.data.data))
    plt.xlim([f_min, 3*f_min])
    plt.savefig('injection_re_h.png')
    plt.clf()
  
  return [Hp, Hc]
  
#############################################################
def InjectTidalWaveform_ChooseFD(m1, m2, S1x=0, S1y=0, S1z=0, S2x=0, S2y=0, S2z=0, Lambda=500, f_min=10.0, f_max=2048.0, f_ref=0.0, deltaF=0.1, approximant=lalsim.SEOBNRv2, distance=1e6*PC_SI, make_plots=False):
  #
  try: approx = lalsim.GetStringFromApproximant(approximant)
  except: raise IOError("Approximant %d not supported" % approximant)
  
  # Initialization of class requires approximant name
  tw = tidalWavs(approx=approx, verbose=False)
  
  Mtotal = m1 + m2
  eta = m1 * m2 / Mtotal**2
  [hp, hc] = tw.getWaveform( Mtotal, eta, S2z, Lambda, distance=distance,\
                             delta_f=deltaF, f_lower=f_min, f_final=f_max )
  Hp = convert_FrequencySeries_to_lalREAL16FrequencySeries( hp )
  Hc = convert_FrequencySeries_to_lalREAL16FrequencySeries( hc )

  if make_plots:
    f = np.arange(Hp.data.length) * Hp.deltaF
    amp_hp = np.abs(Hp.data.data)
    phi_hp = np.unwrap(np.angle(Hp.data.data))
    
    # Make plots of injection
    plt.loglog(f, amp_hp)
    plt.xlabel(r'$f[Hz]$')
    plt.ylabel(r'$|\tilde h|$')
    plt.xlim([f_min, f_max])
    plt.savefig('injection_amplitude.png')
    plt.clf()

    plt.semilogx(f, phi_hp)
    plt.xlabel(r'$f[Hz]$')
    plt.ylabel(r'$\phi[\tilde h]$')
    plt.xlim([f_min, f_max])
    plt.savefig('injection_phase.png')
    plt.clf()

    plt.plot(f, np.real(Hp.data.data))
    plt.xlim([f_min, 3*f_min])
    plt.savefig('injection_re_h.png')
    plt.clf()
  
  return [Hp, Hc]
#############################################################
#############################################################

if __name__ == "__main__":
  # tests
  m1 = 30
  m2 = 20
  chi1 = 0.2
  chi2 = 0.3
  f_min = 10.0
  f_max = 2048.0
  [Hp, Hc]                   = InjectWaveform         (m1, m2, S1z=chi1, S2z=chi2, f_min=f_min, f_max=f_max, deltaF=0.1, approximant=lalsim.SEOBNRv2_ROM_SingleSpin)
  [Hp_ChooseFD, Hc_ChooseFD] = InjectWaveform_ChooseFD(m1, m2, S1z=chi1, S2z=chi2, f_min=f_min, f_max=f_max, deltaF=0.1, approximant=lalsim.SEOBNRv2_ROM_SingleSpin)
  f = np.arange(Hp.data.length) * Hp.deltaF
  assert(Hp.data.length == Hp_ChooseFD.data.length)
  
  plt.loglog(f, np.abs(Hp.data.data))
  plt.loglog(f, np.abs(Hp_ChooseFD.data.data), 'r--')
  plt.xlabel(r'$f[Hz]$')
  plt.ylabel(r'$|\tilde h|$')
  plt.xlim([f_min, f_max])
  plt.show()
  
  # Jolien changes the phasing in SimInspiralFD
  # need fine deltaF to get rid of these disagreements
  phi_H = np.unwrap(np.angle(Hp.data.data))
  phi_H_ChooseFD = np.unwrap(np.angle(Hp_ChooseFD.data.data))
  plt.plot(f, phi_H)
  plt.plot(f, phi_H_ChooseFD, 'r--')
  plt.xlabel(r'$f[Hz]$')
  plt.ylabel(r'$\phi[\tilde h]$')
  plt.xlim([f_min, 30]);
  plt.ylim([-50,50]);
  plt.show()
    
  [c_t, c_phi] = fit_dephasing(Hp, Hp_ChooseFD, f_min=f_min, f_max=512.0, plot=True)
  print '[c_t, c_phi]:', [c_t, c_phi]  
  plt.show()
  
  print 'match:', match_FS(Hp, Hp_ChooseFD, lalsim.SimNoisePSDaLIGOZeroDetHighPower, zpf=5)
  print 'match:', match_FS(Hp, Hc_ChooseFD, lalsim.SimNoisePSDaLIGOZeroDetHighPower, zpf=5)
  print 'match:', match_FS(Hp, Hc, lalsim.SimNoisePSDaLIGOZeroDetHighPower, zpf=5)
  
