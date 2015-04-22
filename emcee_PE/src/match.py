# MP 11/2014 - 04/2015

import numpy as np
from matplotlib import pyplot as plt

def snr(h, psdfun):
  """
  Compute the snr of the FD waveform h
  
  :param h:      COMPLEX16FrequencySeries
  :param psdfun: power spectral density as a function of frequency in Hz
  :param zpf:    zero-padding factor
  """
  n = h.data.length
  deltaF = h.deltaF
  f = deltaF*np.arange(0,h.data.length)
  psd = np.array(map(psdfun, f))
  psd[0] = psd[1] # get rid of psdfun(0) = nan
  habs = np.abs(h.data.data)
  return np.sqrt(4*deltaF * np.dot(habs, habs/psd))

def match_FS(h1, h2, psdfun, zpf=5):
  """
  Compute the match between FD waveforms h1, h2
  
  :param h1, h2: COMPLEX16FrequencySeries
  :param psdfun: power spectral density as a function of frequency in Hz
  :param zpf:    zero-padding factor
  """
  assert(h1.data.length == h2.data.length)
  assert(h1.deltaF == h2.deltaF)
  n = h1.data.length
  deltaF = h1.deltaF
  f = deltaF*np.arange(0,n)
  psd_ratio = psdfun(100) / np.array(map(psdfun, f))
  psd_ratio[0] = psd_ratio[1] # get rid of psdfun(0) = nan
  h1abs = np.abs(h1.data.data)
  h2abs = np.abs(h2.data.data)
  norm1 = np.dot(h1abs, h1abs*psd_ratio)
  norm2 = np.dot(h2abs, h2abs*psd_ratio)
  integrand = h1.data.data * h2.data.data.conj() * psd_ratio # different name!
  #integrand_zp = np.lib.pad(integrand, n*zpf, 'constant', constant_values=0) # zeropad it
  integrand_zp = np.concatenate([np.zeros(n*zpf), integrand, np.zeros(n*zpf)]) # zeropad it, in case we don't have np.lib.pad
  csnr = np.asarray(np.fft.fft(integrand_zp)) # complex snr; numpy.fft = Mma iFFT with our conventions
  return np.max(np.abs(csnr)) / np.sqrt(norm1*norm2)

def match(h1, h2, psdfun, deltaF, zpf=5):
  """
  Compute the match between FD waveforms h1, h2
  
  :param h1, h2: data from frequency series
  :param psdfun: power spectral density as a function of frequency in Hz
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
  integrand = h1 * h2.conj() * psd_ratio # different name!
  #integrand_zp = np.lib.pad(integrand, n*zpf, 'constant', constant_values=0) # zeropad it
  integrand_zp = np.concatenate([np.zeros(n*zpf), integrand, np.zeros(n*zpf)]) # zeropad it, in case we don't have np.lib.pad
  csnr = np.asarray(np.fft.fft(integrand_zp)) # complex snr; numpy.fft = Mma iFFT with our conventions
  return np.max(np.abs(csnr)) / np.sqrt(norm1*norm2)
  
def fit_dephasing(h1, h2, f_min=10, f_max=2048.0, plot=True):
  assert(h1.data.length == h2.data.length)
  assert(h1.deltaF == h2.deltaF)
  n = h1.data.length
  f_in = np.arange(n) * h1.deltaF
  phi1_in = np.unwrap(np.angle(h1.data.data))
  phi2_in = np.unwrap(np.angle(h2.data.data)) # should infer sign from sign of frequency

  h_mask = abs(h1.data.data) > 0
  f_mask = (f_in >= f_min) & (f_in <= f_max)
  mask = h_mask & f_mask

  f = f_in[mask]
  phi1 = phi1_in[mask]
  phi2 = phi2_in[mask]
  dphi = phi1 - phi2

  x = f[::10]
  y = dphi[::10]
  [c_t, c_phi] = np.polyfit(x, y, 1)
  
  if plot:
    plt.semilogx(f, dphi - c_t*f - c_phi)
    plt.xlabel(r'$f[Hz]$')
    plt.ylabel(r'$\Delta\phi[\tilde h] - \mathrm{fit}$')
  
  return [c_t, c_phi]
  
def complex_SNR_FS(h1, h2, psdfun, zpf=5):
  """
  Compute the complex SNR (or correlation function) between FD waveforms h1, h2
  This is the iFFT of the noise weighted inner product.
  
  :param h1, h2: COMPLEX16FrequencySeries
  :param psdfun: power spectral density as a function of frequency in Hz
  :param zpf:    zero-padding factor
  """
  assert(h1.data.length == h2.data.length)
  assert(h1.deltaF == h2.deltaF)
  n = h1.data.length
  deltaF = h1.deltaF
  f = deltaF*np.arange(0,n)
  psd_ratio = psdfun(100) / np.array(map(psdfun, f))
  psd_ratio[0] = psd_ratio[1] # get rid of psdfun(0) = nan
  integrand = h1.data.data * h2.data.data.conj() * psd_ratio # different name!
  #integrand_zp = np.lib.pad(integrand, n*zpf, 'constant', constant_values=0) # zeropad it
  integrand_zp = np.concatenate([np.zeros(n*zpf), integrand, np.zeros(n*zpf)]) # zeropad it, in case we don't have np.lib.pad
  csnr = np.asarray(np.fft.fft(integrand_zp)) # complex snr; numpy.fft = Mma iFFT with our conventions
  return csnr
