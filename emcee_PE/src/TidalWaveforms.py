#!/usr/bin/nv python

import os
import sys
#import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

import lal
from pycbc import pnutils
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
  Returns waveform windowing_function w_{f0,d}(f), for the input frequencies
  """
  return 0.5*(1. + sgn*np.tanh(4.*(f - f0)/d))

def lorentzian(f, f0, d): return d**2 / ((f - f0)**2 + d**2 / 4.)
  
class PNcoeffs():
  def __init__(self, mass1=5, mass2=5, spin1z=0, spin2z=0, verbose=True):
    """
    This class is for storing all PN coefficients, as well for solving for 
    orbital quantities.
    """
    if not hasattr(self, 'verbose'): self.verbose = verbose
    
    self.total_mass = mass1 + mass2
    self.eta = mass1 * mass2 / self.total_mass**2
    self.spin1z = spin1z
    self.spin2z = spin2z
    self.chi = (mass1 * spin1z + mass2 * spin2z)/self.total_mass
    
    self.coeffs = {}
    self.initialize_coefficients_fd()
    return
  #
  def initialize_coefficients_fd(self):
    """
    Initialize PN coefficients. Currently implemented are
    - 3PN amplitude corrections, for aligned-spins. given by Eq.A5 of 1005.3306
    """
    eta, chi1, chi2, chi = self.eta, self.spin1z, self.spin2z, self.chi
    pi2 = np.pi * np.pi
    eta2 = eta * eta
    piM = self.total_mass * np.pi * lal.MTSUN_SI
    chisum = chi + chi    ## This is the chisum in phenomC, is this right?
    chiprod = chi * chi   ## This is the chiprod in phenomC, is this right?
    chi2 = chi * chi      ## This is the chi**2 in phenomC, is this right?
    
    # Amplitude coefficients
    ampc = {}
    ampc['AN'] = 8 * eta * np.sqrt(np.pi/5.)
    ampc['A0'] = 1.
    ampc['A1'] = 0.
    ampc['A2'] = (-107. + 55.*eta)/42.
    ampc['A3'] = 2.*np.pi - 4.*chi/3. + 2.*eta*chisum/3.
    ampc['A4'] = -(2173./1512.) - eta*(1069./216. - 2.*chiprod) + 2047.*eta2/1512.
    ampc['A5'] = -(107*np.pi/21.) + eta*(34.*np.pi/21. - 24j)
    ampc['A6'] = 27027409./646800. - 856.*np.euler_gamma/105. +\
                  np.pi*428.0j/105. + 2.*pi2/3. +\
                  eta*(41.*pi2/96. - 278185./33264.) - 20261.*eta2/2772. +\
                  114635*eta*eta2/99792. - 428.*np.log(16.)/105.
    ampc['A6log'] = -428./105.
    
    # Phase coefficients
    phsc = {}
    phsc['B0'] = 1.
    
    # dx/dt coefficients
    # Coefficients to calculate xdot, that comes in the fourier amplitude 
    xdc = {}
    xdc['aN'] = 64.*eta/5.
    xdc['a2'] = -7.43/3.36 - 11.*eta/4.
    xdc['a3'] = 4.*np.pi - 11.3*chi/1.2 + 19.*eta*chisum/6.
    xdc['a4'] = 3.4103/1.8144 + 5*chi2 + eta*(13.661/2.016 - chiprod/8.) + 5.9*eta2/1.8
    xdc['a5'] = -np.pi*(41.59/6.72 + 189.*eta/8.) - chi*(31.571/1.008 - 116.5*eta/2.4) +\
          chisum*(21.863*eta/1.008 - 79.*eta2/6.) - 3*chi*chi2/4. +\
          9.*eta*chi*chiprod/4.
    xdc['a6'] = 164.47322263/1.39708800 - 17.12*np.euler_gamma/1.05 +\
          16.*np.pi*np.pi/3 - 8.56*np.log(16.)/1.05 +\
          eta*(45.1*np.pi*np.pi/4.8 - 561.98689/2.17728) +\
          5.41*eta2/8.96 - 5.605*eta*eta2/2.592 - 80.*np.pi*chi/3. +\
          eta*chisum*(20.*np.pi/3. - 113.5*chi/3.6) +\
          chi2*(64.153/1.008 - 45.7*eta/3.6) -\
          chiprod*(7.87*eta/1.44 - 30.37*eta2/1.44)

    xdc['a6log'] = -856./105.
    
    xdc['a7'] = -np.pi*(4.415/4.032 - 358.675*eta/6.048 - 91.495*eta2/1.512) -\
          chi*(252.9407/2.7216 - 845.827*eta/6.048 + 415.51*eta2/8.64) +\
          chisum*(158.0239*eta/5.4432 - 451.597*eta2/6.048 + 20.45*eta2*eta/4.32 +\
            107.*eta*chi2/6. - 5.*eta2*chiprod/24.) +\
          12.*np.pi*chi2 - chi2*chi*(150.5/2.4 + eta/8.) +\
          chi*chiprod*(10.1*eta/2.4 + 3.*eta2/8.)

    self.coeffs['tdamplitude-fd'] = ampc
    self.coeffs['tdphase-fd'] = phsc
    self.coeffs['xdot-fd'] = xdc    
    return
  #
  def GetWaveformTDAmplitudeFD(self, freqs, total_mass, distance):
    """
    freqs : Hz
    total_mass : solar mass
    distance : Mpc
    """
    if hasattr(self, 'coeffs') and hasattr(self.coeffs, 'tdamplitude-fd'):
      self.initialize_coefficients_fd()
    ampc = self.coeffs['tdamplitude-fd']
    
    v = (np.pi * total_mass * lal.MTSUN_SI * freqs)**(1./3.)
    v2 = v*v
    v3 = v*v2
    v4 = v*v3
    v5 = v*v4
    v6 = v*v5
    logv = np.log(v)
    
    s = 0 + 0j
    s += ampc['A0'] + ampc['A1']*v + ampc['A2']*v2 + ampc['A3']*v3
    s += ampc['A4']*v4 + ampc['A5']*v5 + ampc['A6']*v6
    s += ampc['A6log']*logv*v6
    s *= ampc['leading']*v2*total_mass*lal.MRSUN_SI/(distance*1.e6*lal.PC_SI)
    return s
  #
  def GetWaveformAmplitudeFD(self, freqs, total_mass, distance):
    """
    freqs : Hz
    total_mass : solar mass
    distance : Parsec
    
    Using Eq.3.9, 3.10, 3.14, 3.15 from 1005.3306
    A(f) comprises of two parts, A(x) and \sqrt{\pi / \dot{\omega}}
    First we compute \dot{\omega}
    Then we compute A(x)
    """
    if not (hasattr(self, 'coeffs') and hasattr(self.coeffs, 'amplitude-fd') \
              and hasattr(self, 'xdot-fd')):
      self.initialize_coefficients_fd()
    ampc = self.coeffs['tdamplitude-fd']
    xdc  = self.coeffs['xdot-fd']
    
    amp0 = 2. * np.sqrt(5. / (64.*np.pi)) * total_mass * lal.MRSUN_SI * \
                      total_mass * lal.MTSUN_SI / (distance * lal.PC_SI)
    
    v = (np.pi * total_mass * lal.MTSUN_SI * freqs)**(1./3.)
    v2 = v*v
    v3 = v*v2
    v4 = v*v3
    v5 = v*v4
    v6 = v*v5
    v7 = v*v6
    v10= v7*v3
    logv = np.log(v)
    
    # Get the amplitude 
    xdot = 0 + 0j
    xdot += 1. + xdc['a2']*v2 + xdc['a3']*v3 + xdc['a4']*v4
    xdot += xdc['a5']*v5 + (xdc['a6'] + xdc['a6log']*2.*logv)*v6
    xdot += xdc['a7'] * v7
    xdot *= (xdc['aN'] * v10)

    #/* Following Emma's code, take only the absolute value of omegaDot, when
    #* computing the amplitude */
    omgdot = 1.5*v*xdot;
    ampfac = np.sqrt(np.abs(np.pi/omgdot));
    
    # Get the real and imaginary part of A(x)
    s = 0 + 0j
    s += ampc['A0'] + ampc['A1']*v + ampc['A2']*v2 + ampc['A3']*v3
    s += ampc['A4']*v4 + ampc['A5']*v5 + (ampc['A6'] + ampc['A6log']*logv)*v6
    s *= (amp0 * ampfac * ampc['AN'] * v2)
    
    # Following Emma's code, we take the absolute part of the complex SPA
    # amplitude, and keep that as the amplitude
    aPN = np.abs(s)
    
    return aPN, amp0
  #
  def GetWaveformPhase(self):
    return

class tidalWavsFP(PNcoeffs):
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
      P3 : arXiv:0512160
      """
      PNcoeffs.__init__(self, mass1=massBH, mass2=massNS, spin1z=spinBH, \
                        spin2z=0, verbose=verbose)
      
      if massBH <= 0 or massNS <= 0 or spinBH > 1 or spinBH < -1 or \
          massNS_B < 0 or radiusNS <= 0:
        raise IOError("Provide correct input to waveform generation please")
        
      self.massBH = massBH
      self.spinBH = spinBH
      self.massNS = massNS
      
      # The following two have to be determined from the EoS ..[TODO]
      self.radiusNS = radiusNS
      self.massNS_B = massNS_B
      self.Lambda   = (radiusNS / massNS)**5
      
      
      self.mtotal = self.massNS + self.massBH
      self.eta = self.massNS * self.massBH / self.mtotal**2
      
      self.f_lower = f_lower
      self.delta_f = delta_f
      
      self.verbose = verbose
      self.approx  = approx
      #
      # Initialize constants
      self.get_calibrated_parameters()
      
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
      total_mass = self.mtotal 
      mf, af = self.get_final_blackhole_mass_spin()
      
      sgn = +1 # depends on the component pre-merger BH's spin
      if self.spinBH != 0: sgn = self.spinBH / np.abs(self.spinBH)
      
      self.mfTide = sgn / (np.pi * (af*mf + np.sqrt(rTide**3 / mf)))
      self.fTide  = self.mfTide / (total_mass * lal.MTSUN_SI)
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
    def get_ringdown_quality_frequency(self):
      """
      Calculates the frequency and quality factor of ringdown of the post-merger
      BH. Expressions used are given in table VIII of P3.
      """
      if hasattr(self, 'fRD') and hasattr(self, 'QRD'):
        return self.fRD, self.QRD
      
      mf = self.massBH_final()
      af = self.spinBH_final()
      
      f1, f2, f3 = [1.5251, -1.1568, +0.1292]
      q1, q2, q3 = [0.7000, +1.4187, -0.4990]
      
      j = np.abs(af) # WHAT IS THIS ?
      
      self.mfRD = f1 + f2 * (1. - j)**f3
      self.fRD  = self.mfRD / (mf * lal.MTSUN_SI)
      self.QRD  = q1 + q2 * (1. - j)**q3
      
      return self.fRD, self.QRD
    #
    def freqRD(self):
      """
      Calculates the frequency of ringdown of the post-merger
      BH. Expressions used are given in table VIII of P3.
      """
      if hasattr(self, 'fRD'): return self.fRD
      
      frd, _ = self.get_ringdown_quality_frequency()
      return frd
    #
    def QualityRD(self):
      """
      Calculates the quality factor of ringdown of the post-merger
      BH. Expressions used are given in table VIII of P3.
      """
      if hasattr(self, 'QRD'): return self.QRD
      
      _, qrd = self.get_ringdown_quality_frequency()
      return qrd      
    #
    def get_calibrated_parameters(self):
      """
      Calculate the 
      - e_tide
      - e_ins
      - sigma_tide
      - delta2p
      coefficients, as per Eq.16-27 of P1
      """
      if hasattr(self, 'coeffs') and hasattr(self.coeffs, 'TidalModel'):
          return self.coeffs['TidalModel']
      
      eta     = self.eta
      sBH     = self.spinBH
      
      d       = 0.015 # Value from PhenomC model
      C       = self.massNS / self.radiusNS
      q       = self.massBH / self.massNS
      fRD     = self.freqRD()
      fTide   = self.freqTide()
      mbtorus = self.massBHTorus()
      
      if fTide >= fRD and mbtorus == 0:
        print "NON-DISRUPTIVE MERGER"
        e_ins = 1
        
        fdiffratio = (fTide - fRD) / fRD
        x1, d1 = [-0.0796251, 0.0801192]
        xND = fdiffratio**2 - 0.571505*C - 0.00508451*sBH
        e_tide = 2 * windowing_function(xND, x1, d1, sgn=1)
        
        x2, d2 = [-0.206465, 0.226844]
        xNDp = fdiffratio**2 - 0.657424*C - 0.0259977*sBH
        sigma_tide = 2. * windowing_function(xNDp, x2, d2, sgn=-1)
        
        A, x3, d3 = [1.62496, 0.0188092, 0.338737]
        delta2p = A * windowing_function(fdiffratio, x3, d3, sgn=-1)
        
        c1 = c2 = c3 = 1
        f0 = f1 = f2 = f3 = fRD
        d1 = d2 = d3 = d + sigma_tide
        
        coeffs = {'c1' : c1, 'c2' : c2, 'c3' : c3, 
                  'f1' : f1, 'f2' : f2, 'f3' : f3, 
                  'd1' : d1, 'd2' : d2, 'd3' : d3, 
                  'e_ins' : e_ins, 'e_tide' : e_tide, 
                  'sigma_tide' : sigma_tide, 'delta2p' : delta2p}
        #
      elif fTide < fRD and mbtorus > 0:
        print "DISRUPTIVE MERGER"
        a1, b1 = [1.29971, -1.61724]
        xD = (mbtorus / self.massNS_B) + 0.424912*C + 0.363604 * eta**0.5 - 0.0605591*sBH
        e_ins = a1 + b1 * xD
        
        e_tide = None
        f0 = fTide
        
        a2, b2 = 0.137722, -0.293237
        xDp = (mbtorus / self.massNS_B) - 0.132754*C + 0.576669 * eta**0.5 - \
                0.0603749*sBH - 0.0601185*sBH*sBH - 0.0729134 * sBH**3
        sigma_tide = a2 + b2 * xDp
        delta2p = None # DUMMY VALUE
        
        c1 = c2 = 1
        c3 = 0
        f1 = e_ins * fTide
        f2 = fTide
        d1 = d2 = d + sigma_tide
        d3 = f3 = None
        
        coeffs = {'c1' : c1, 'c2' : c2, 'c3' : c3, 
                  'f1' : f1, 'f2' : f2, 'f3' : f3, 
                  'd1' : d1, 'd2' : d2, 'd3' : d3, 
                  'e_ins' : e_ins, 'e_tide' : e_tide, 
                  'sigma_tide' : sigma_tide, 'delta2p' : delta2p}
        #
      elif fTide < fRD and mbtorus == 0:
        print "MILDLY DISRUPTIVE MERGER :CHECKME"
        a1, b1 = [1.29971, -1.61724]
        xD = (mbtorus / self.massNS_B) + 0.424912*C + 0.363604 * eta**0.5 - 0.0605591*sBH
        e_ins = a1 + b1 * xD
        
        e_tide = None
        
        x2, d2 = [-0.206465, 0.226844]
        xNDp = fdiffratio**2 - 0.657424*C - 0.0259977*sBH
        sigma_tide = 2. * windowing_function(xNDp, x2, d2, sgn=-1) / 2.
        
        a2, b2 = 0.137722, -0.293237
        xDp = (mbtorus / self.massNS_B) - 0.132754*C + 0.576669 * eta**0.5 - \
                0.0603749*sBH - 0.0601185*sBH*sBH - 0.0729134 * sBH**3
        sigma_tide += (a2 + b2 * xDp)/2.
        
        delta2p = None
        
        c1 = c2 = 1
        c3 = 0
        f1 = (1. - 1./q) * fRD + e_ins*fTide / q
        f2 = (1. - 1./q) * fRD + fTide / q
        f3 = d3 = None
        d1 = d2 = d + sigma_tide
        
        coeffs = {'c1' : c1, 'c2' : c2, 'c3' : c3, 
                  'f1' : f1, 'f2' : f2, 'f3' : f3, 
                  'd1' : d1, 'd2' : d2, 'd3' : d3, 
                  'e_ins' : e_ins, 'e_tide' : e_tide, 
                  'sigma_tide' : sigma_tide, 'delta2p' : delta2p}
        #
      elif fTide >= fRD and mbtorus > 0:
        print "MILDLY DISRUPTIVE MERGER WITH A TORUS"
        a1, b1 = [1.29971, -1.61724]
        xD = (mbtorus / self.massNS_B) + 0.424912*C + 0.363604 * eta**0.5 - 0.0605591*sBH
        e_ins = a1 + b1 * xD
        
        fdiffratio = (fTide - fRD) / fRD
        x1, d1 = [-0.0796251, 0.0801192]
        xND = fdiffratio**2 - 0.571505*C - 0.00508451*sBH
        e_tide = 2 * windowing_function(xND, x1, d1, sgn=1)
        
        fdiffratio = (fTide - fRD) / fRD
        x2, d2 = [-0.206465, 0.226844]
        xNDp = fdiffratio**2 - 0.657424*C - 0.0259977*sBH
        sigma_tide = 2. * windowing_function(xNDp, x2, d2, sgn=-1)
        
        A, x3, d3 = [1.62496, 0.0188092, 0.338737]
        delta2p = A * windowing_function(fdiffratio, x3, d3, sgn=-1)
        
        c1 = c2 = c3 = 1
        f1 = f2 = e_ins * fRD
        f3 = fRD
        d1 = d2 = d3 = d + sigma_tide
        
        coeffs = {'c1' : c1, 'c2' : c2, 'c3' : c3, 
                  'f1' : f1, 'f2' : f2, 'f3' : f3, 
                  'd1' : d1, 'd2' : d2, 'd3' : d3, 
                  'e_ins' : e_ins, 'e_tide' : e_tide, 
                  'sigma_tide' : sigma_tide, 'delta2p' : delta2p}        
      #  
      if not hasattr(self, 'coeffs'): self.coeffs = {}
      self.coeffs['TidalModel'] = coeffs
      return coeffs
    #
    def getWaveformAmplitude(self, mtotal=None, eta=None, sBH=None, Lambda=None,\
                    distance=1e6,\
                    f_lower=15., f_final=4096., \
                    delta_t=1./8192., delta_f=1./256, tidal=True):
      """
      Compute the amplitude of a BHNS inspiral-merger-ringdown, as a function
      of emitted GW frequency.
      
      APhen(f) =             c1 APN(f) a1 winPN(f1,d1) 
                + 1.25 gamma1 c2 f^5/6 a2 winM(f2,d2) 
                +            c3 ARD(f) a3 winRD(f3, d3)
      
      where :
      APN(f) is the 3.5PN expansion of the amplitude
      winPN(f) is a window that ends at e_ins * f0, width = d + sigma_tide
      winM(f)  is a window which ends at        f0, width = d + sigma_tide
      winRD(f) is a window which starts at      f0, width = d + sigma_tide
      
      and
      
      ARD(f) = e_tide * delta1 * Lorentzian(f, fRD, delta2' fRD / Q) * f^-7/6
      
      Please go through Sec.IV of P1 for all details.
      """
      if mtotal is None: mtotal = self.mtotal
      if eta is None: eta = self.eta
      if sBH is None: sBH = self.spinBH
      if Lambda is None: Lambda = self.lambdaNS
      
      m1, m2 = pnutils.mtotal_eta_to_mass1_mass2(mtotal, eta)
      
      # Calculate gamma1 and delta1
      chi = sBH * m1 / mtotal
      gamma1 = 4.149*chi - 4.07*chi*chi - 87.52*eta*chi - 48.97*eta + 666.5*eta*eta
      delta1 = -5.472e-2*chi + 2.094e-2*chi*chi + 0.3554*eta*chi + 0.1151*eta + 0.964*eta*eta
      
      # Calculate Mass of the torus around the BH
      mbtorus = self.massBHTorus()
      
      # Calculate the mass and spin of merger BH
      mf, chif = self.get_final_blackhole_mass_spin()
      
      # Calculate RD freq and Quality factor
      fRD, QRD = self.get_ringdown_quality_frequency()
      
      # Calculate frequency of the onset of mass-shedding
      fTide = self.freqTide()
      
      # Get calibrated tidal parameters, depending on if the NS disrupts
      coeffs = self.get_calibrated_parameters()
      
      # Get frequency range
      N = int(np.round( 1./delta_t/delta_f ))
      frequencies = np.arange(N/2 + 1) * delta_f
      
      # Calculate APN
      APN, amp0 = self.GetWaveformAmplitudeFD(frequencies, mtotal, distance)
      APN *= windowing_function(frequencies, coeffs['f1'], coeffs['d1'], sgn=-1)
      
      # Calculate AM
      AM = 0
      if coeffs['c2'] != 0:
        AM = 1.25 * gamma1 * coeffs['c2'] * frequencies**(5./6.) 
        AM *= windowing_function(frequencies, coeffs['f2'], coeffs['d2'], sgn=-1)
      
      # Calculate ARD
      ARD = 0
      if coeffs['c3'] != 0:
        ARD = coeffs['e_tide'] * delta1 * \
                lorentzian(frequencies, fRD, coeffs['delta2p'] * fRD/QRD) * \
                frequencies**(-7./6.)
        ARD *= windowing_function(frequencies, coeffs['f3'], coeffs['d3'])
      
      # Combine
      APhen = FrequencySeries(APN + amp0*(AM + ARD),\
                        delta_f=delta_f, epoch=-f_final, copy=True)
      
      return APhen



































