#!/usr/bin/nv python

import os
import sys
#import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

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
    def tidalCorrectionAmplitude(self, mf, eta, sBH, tidalLambda, mfA=0.01):
        try: temp = len(mf)
        except: mf = np.array([mf])
        # Assume input frequencies, if an array, are ordered
        #if mf <= mfA: return 1
        mask = mf <= mfA
        AmpLowF = np.ones(len( np.nonzero(mask)[0] ))
        mf = mf[np.invert(mask)]
        #
        if len(mf) == 0: return AmpLowF
        # Impose cutoffs on mass-ratio, and BH spins
        if eta < 6./49.: 
            print "Eta = ", eta, 6./49.
            raise IOError("Eta too small")
        if sBH > 0.75: raise IOError("BH spin too large")
        if sBH < -0.75: raise IOError("BH spin too small")
        # Generate the amplitude factor
        C = np.exp(self.b0 + self.b1*eta + self.b2 *sBH) \
            + tidalLambda * np.exp(self.c0 + self.c1*eta + self.c2*sBH)
        D = 3.
        B = C * (mf - mfA)**D
        return np.append( AmpLowF, np.exp(-eta * tidalLambda * B))
    #
    def tidalPNPhase(self, mf, eta, tidalLambda):
        
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
        return psiT
    #
    def tidalPNPhaseDeriv(self, mf, eta, tidalLambda):
        
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
        DpsiT = np.pi * (5.*a0 * pimf3rd**2 + 7.*a1 * pimf3rd**4)/(128.*eta)
        return DpsiT
    #
    def tidalCorrectionPhase(self, mf, eta, sBH, tidalLambda,\
                             mfP=0.02, inspiral=True):
        #
        try: len(mf)
        except TypeError: mf = np.array([mf])
        # Assume input frequencies, if an array, are ordered
        #if mf <= mfP: return self.tidalPNPhase(mf, eta, tidalLambda)
        mask = mf <= mfP
        PhsLowF = self.tidalPNPhase(mf[mask], eta, tidalLambda)
        mf = mf[np.invert(mask)]
        #
        if len(mf) == 0: return PhsLowF
        # 
        if inspiral:
            psiT = self.tidalPNPhase(mfP, eta, tidalLambda)
            DpsiT= (mf - mfP) * self.tidalPNPhaseDeriv(mfP, eta, tidalLambda)         
        # Now compute the phenomenological term
        G = np.exp(self.g0 + self.g1*eta + self.g2*sBH +\
                   self.g3*eta*sBH)
        H = 5./3.
        E = G * (mf - mfP)**H
        psiFit = (eta * tidalLambda * E)
        # Final phase
        psi = psiT + DpsiT - psiFit
        return np.append( PhsLowF, psi)
    #
    def getWaveform(self, M, eta, sBH, Lambda, distance=1e6*lal.PC_SI, \
                    f_lower=15., f_final=4096., \
                    delta_t=1./8192., delta_f=1./256, tidal=True):
        if self.approx in fd_approximants():
            m1, m2 = self.mtot_eta_to_m1_m2(M, eta)
            hp, hc = get_fd_waveform(approximant=self.approx,\
                        mass1=m1, mass2=m2, spin1z=sBH, spin2z=0,\
                        distance=distance, \
                        f_lower=f_lower, f_final=f_final, delta_f=delta_f)
        else: raise IOError("Approx not supported")
        if not tidal or Lambda==0:
            if self.verbose:
                print "Returning WITHOUT tidal corrections"
                tid = int(0.1/M/lal.MTSUN_SI / delta_f)
                print hc[tid], hp[tid]
            return hp, hc
        # Tidal corrections to be incorporated
        freqs =  np.array(M * lal.MTSUN_SI * hp.sample_frequencies.data)
        hpd, hcd = hp.data, hc.data
        #
        ampC = self.tidalCorrectionAmplitude(freqs, eta, sBH, Lambda)
        phsC = self.tidalCorrectionPhase(freqs, eta, sBH, Lambda)
        Corr = np.cos(phsC) - 1j * np.sin(phsC)
        #
        #ampC = np.array([self.tidalCorrectionAmplitude(mf, eta, sBH, Lambda) for mf in freqs])
        #phsC = np.array([self.tidalCorrectionPhase(mf, eta, sBH, Lambda) for mf in freqs])
        #Corr = np.array([np.complex(np.cos(phsC[ii]),-1*np.sin(phsC[ii])) for ii in range(len(phsC))])
        Corr = Corr * ampC
        hp = FrequencySeries(hp * Corr, delta_f=delta_f, epoch=hp._epoch, dtype=hp.dtype, copy=True)
        hc = FrequencySeries(hc * Corr, delta_f=delta_f, epoch=hp._epoch, dtype=hp.dtype, copy=True)
        if self.verbose:
            tid = int(0.1/M/lal.MTSUN_SI / delta_f)
            print ampC[tid], phsC[tid], Corr[tid]
            print hc[tid], hp[tid]
        return hp, hc
    #}}}


def windowing_function(f, f0, d, sgn=+1):
  """
  Returns waveform windowing_function, for the input frequencies
  """
  return 0.5*(1. + sgn*np.tanh(4.*(f - f0)/d))

class PNcoeffs():
  def __init__(verbose=True):
    """
    This class is for storing all PN coefficients, as well for solving for 
    orbital quantities.
    """
    self.verbose = verbose
    self.initialize_coefficients_fd()
    return
  #
  def initialize_coefficients_fd(self):
    return
  #
  def GetWaveformPhase(self):
    return
  #
  def GetWaveformAmplitude(self):
    return

class tidalWavsFP():
    #{{{
    def __init__(self, approx='IMRPhenomC',\
          massBH = -1, spinBH = 2,\
          massNS_B = -1, massNS = -1, radiusNS = 0, \
          f_lower = 15., delta_f = 1./256, \
          verbose=True):
      """
      This class returns the waveform generated using the model in P1.
      
      References:
      P1 : arXiv:1509.00512
      P2 : arXiv:1311.5931
      """
      if massBH <= 0 or massNS <= 0 or spinBH > 1 or spinBH < -1 or \
          massNS_B < 0 or radiusNS <= 0:
        raise IOError("Provide correct input to waveform generation please")
        
      self.massBH = massBH
      self.spinBH = spinBH
      
      self.massNS = massNS
      # The following two have to be determined from the EoS ..
      self.radiusNS= radiusNS
      self.massNS_B = massNS_B
      
      self.mtotal = self.massNS + self.massBH
      self.eta = self.massNS * self.massBH / self.mtotal**2
      
      self.f_lower = f_lower
      self.delta_f = delta_f
      
      self.verbose = verbose
      self.approx  = approx
      #
      # Initialize constants
      return
    #
    # Compute intermediate quantities
    #
    def rISCO(self, spin=None):
      """
      This function calculates the ISCO radius, as per Eq.12 of Ref P1
      """
      if spin is None or spin < -1 or spin > 1:
        raise IOError("Input spin to rISCO should be in [-1,1]")
      #
      a = spin
      sgn = 1
      if a != 0: sgn = a / np.abs(a)
      
      Z1 = 1. + (1. - a*a)**(1./3.) * ((1. + a)**(1./3.) + (1. - a)**(1./3.))
      Z2 = np.sqrt(3. * a * a + Z1 * Z1)
      rIsco = 3. + Z2 - sgn * np.sqrt((3. - Z1) * (3. + Z1 + 2.*Z2))
      return rIsco
    #
    def rISCOi(self):
      """
      Return the initial BH component's ISCO radius
      """
      if hasattr(self, 'rISCO_i'): return self.rISCO_i
      self.rISCO_i = self.rISCO(spin = self.spinBH)
      return self.rISCO_i
    #
    def rISCOf(self):
      """
      Return the final BH's ISCO radius
      """
      if hasattr(self, 'rISCO_f'): return self.rISCO_f
      if hasattr(self, 'spinBH_f'):
        self.rISCO_f = self.rISCO(spin = self.spinBH_f)
      else:
        self.rISCO_f = self.rISCO(spin = self.spinBH_final())
      return self.rISCO_f
    #
    def radiusTide(self):
      """
      This function returns the radial separation at which mass-shedding 
      will begin for the binary.
      """
      if hasattr(self, 'rTide'): return self.rTide
      # Setup to solve Eq.8 of P1 to get zeta_tide
      def root_fun(zeta_tide):
        mu = self.massBH / self.radiusNS
        lhs = self.massNS * zeta_tide**3 / self.massBH
        rhsnum = 3.*( zeta_tide**2 - 2.*mu*zeta_tide + mu**2 * self.spinBH**2)
        rhsden = zeta_tide**2 - 3.*mu*zeta_tide +\
                  2.*self.spinBH*np.sqrt(mu**3 * zeta_tide)
        return (lhs - (rhsnum/rhsden))
      
      result = minimize_scalar(root_fun, method='Bounded', bounds=(2, 10), \
                                  options={'disp' : True})
      zeta_tide = result.x
      self.rTide = \
        zeta_tide * self.radiusNS * (1. - 2.*self.massNS / self.radiusNS)
      # Return the mass-shedding radius
      return self.rTide
    #
    def freqTide(self):
      """
      This function returns the tidal frequency, where mass-shedding begins,
      calculated using Eq.10 of P1
      """
      if hasattr(self, 'fTide'): return self.fTide      
      
      rTide  = self.radiusTide()      
      mf, af = self.get_final_blackhole_mass_spin()
      
      sgn = +1 # depends on the component pre-merger BH's spin
      if self.spinBH != 0: sgn = self.spinBH / np.abs(self.spinBH)
      
      self.fTide = sgn / (np.pi * (af*mf + np.sqrt(rTide**3 / mf)))
      return self.fTide
    #
    def massBHTorus(self):
      """
      This function returns the mass of the torus formed around the post-merger
      BH, using Eq.11 of P1
      """
      if hasattr(self, 'massBH_Torus'): return self.massBH_Torus
      
      rIscoI = self.rISCOi()
      rTide  = self.radiusTide()
      self.massBH_Torus = \
                  (self.massNS_B / self.radiusNS) * (0.296*rTide - 0.171*rIscoI)
      return self.massBH_Torus
    #
    def efunc(self, r, a):
      """
      Return the fnction defined in Eq.3 of P2
      """
      sgn = +1
      if a != 0: sgn = a / np.abs(a)
      
      num = r*r - 2.*r + sgn*a*np.sqrt(r)
      den = r * np.sqrt( r*r - 3.*r + sgn*2.*a*np.sqrt(r) )
      return num/den
    #
    def lzfunc(self, r, a):
      """
      Return the fnction defined in Eq.2 of P2
      """
      sgn = +1
      if a != 0: sgn = a / np.abs(a)
      
      num = r*r - sgn*2.*a*np.sqrt(r) + a*a
      den = np.sqrt( r * (r*r - 3.*r + sgn*2.*a*np.sqrt(r)) )
      return sgn * num / den
    #
    def fetafunc(self, eta):
      """
      Return the piecewise fnction defined in Eq.5 of P2
      """
      if eta > 0.25 or eta < 0: raise IOError("Eta must be in [0,0.25]")
      if eta <= 0.16: return 0
      if eta >= (2./9.): return 1
      return 0.5 * (1. - np.cos(np.pi * (eta - 0.16)/((2./9.) - 0.16)))      
    #
    def get_final_blackhole_mass_spin(self):
      """
      Calculates and returns the mass and spin of the post-merger BH.
      This uses Eq.1-6 of P2.
      """
      if not hasattr(self, 'spinBH_f'):
        # Set up the solver to solve Eq.4 of P2
        def root_fun(af):
          feta = self.fetafunc(self.eta)
          Mbtorus = self.massBHTorus()
          ei = self.efunc( self.rISCOi(), self.spinBH )
          ef = self.efunc( self.rISCO(spin=af), af )
          rhsnum = self.spinBH * self.massBH**2 + \
                  self.lzfunc( self.rISCO(spin=af), af ) * self.massBH * \
                  ((1. - feta) * self.massNS + feta*self.massNS_B - Mbtorus)
          rhsden = (self.mtotal*(1. - (1.-ei)*self.eta) - ef * Mbtorus)**2
          return (af - (rhsnum / rhsden))
        #
        result = minimize_scalar(root_fun, method='Bounded', bounds=(-1,1))
        self.spinBH_f = result.x
      #    
      if not hasattr(self, 'massBH_f'):
        rIscoF = self.rISCOf()
        ei = self.efunc( self.rISCOi(), self.spinBH )
        ef = self.efunc(        rIscoF, self.spinBH_f )
        Mbtorus = self.massBHTorus()
        self.massBH_f = \
          self.mtotal * (1. - (1. - ei)*self.eta) - ef * Mbtorus
      #
      return [self.massBH_f, self.spinBH_f]
    #
    def massBH_final(self):
      """
      Calculates and returns the mass of the post-merger BH.
      This uses Eq.1-6 of P2.
      """
      if hasattr(self, 'massBH_f'): return self.massBH_f
      mf, _ = self.get_final_blackhole_mass_spin()
      return mf
    #
    def spinBH_final(self):
      """
      Calculates and returns the spin of the post-merger BH.
      This uses Eq.1-6 of P2.
      """
      if hasattr(self, 'spinBH_f'): return self.spinBH_f
      _, af = self.get_final_blackhole_mass_spin()
      return af
    #
    def freqRD(self):
      if hasattr(self, 'fRD'): return self.fRD
      return
    #
    def getWaveformAmplitude(self, M, eta, sBH, Lambda, distance=1e6*lal.PC_SI,\
                    f_lower=15., f_final=4096., \
                    delta_t=1./8192., delta_f=1./256, tidal=True):
      """
      Compute the amplitude of a BHNS inspiral-merger-ringdown, as a function
      of emitted GW frequency.
      """
      return
