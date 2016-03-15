# src/lscsoft/lalsuite/pylal/pylal/bayespputils.py

from scipy import signal

def autocorrelation(series):
    """Returns an estimate of the autocorrelation function of a given
    series.  Returns only the positive-lag portion of the ACF,
    normalized so that the zero-th element is 1."""
    x = series - np.mean(series)
    y = np.conj(x[::-1])

    acf = np.fft.ifftshift(signal.fftconvolve(y,x,mode='full'))

    N = series.shape[0]

    acf = acf[0:N]

    return acf/acf[0]

def autocorrelation_length_estimate(series, acf=None, M=5, K=2):
    """Attempts to find a self-consistent estimate of the
    autocorrelation length of a given series.

    If C(tau) is the autocorrelation function (normalized so C(0) = 1,
    for example from the autocorrelation procedure in this module),
    then the autocorrelation length is the smallest s such that

    1 + 2*C(1) + 2*C(2) + ... + 2*C(M*s) < s

    In words: the autocorrelation length is the shortest length so
    that the sum of the autocorrelation function is smaller than that
    length over a window of M times that length.

    The maximum window length is restricted to be len(series)/K as a
    safety precaution against relying on data near the extreme of the
    lags in the ACF, where there is a lot of noise.  Note that this
    implies that the series must be at least M*K*s samples long in
    order to get a reliable estimate of the ACL.

    If no such s can be found, raises ACLError; in this case it is
    likely that the series is too short relative to its true
    autocorrelation length to obtain a consistent ACL estimate."""

    if acf is None:
      acf = autocorrelation(series)
    acf[1:] *= 2.0

    imax = int(acf.shape[0]/K)

    # Cumulative sum and ACL length associated with each window
    cacf = np.cumsum(acf)
    s = np.arange(1, cacf.shape[0]+1)/float(M)

    # Find all places where cumulative sum over window is smaller than
    # associated ACL.
    estimates = np.flatnonzero(cacf[:imax] < s[:imax])

    if estimates.shape[0] > 0:
        # Return the first index where cumulative sum is smaller than
        # ACL associated with that index's window
        return s[estimates[0]]
    else:
        # Cannot find self-consistent ACL estimate.
        #print 'autocorrelation length too short for consistent estimate'
        return -1

def effectiveSampleSize(samples, Nskip=1):
    """
    Compute the effective sample size, calculating the ACL using only
    the second half of the samples to avoid ACL overestimation due to
    chains equilibrating after adaptation.
    """
    N = len(samples)
    acf = autocorrelation(samples[N/2:])
    acl = autocorrelation_length_estimate(samples[N/2:], acf=acf)
    if acl == -1:
        acl = N
    Neffective = np.floor(N/acl)
    acl *= Nskip
    return (Neffective, acl, acf)

def samples_from_chain(chain, loglike, i, param=0, burnin=500):
    samples_ok = chain[i, burnin:, param]
    loglike_ok = loglike[burnin:,i]
    sel = np.isfinite(loglike_ok) # discard samples with logL = -inf which are outside of the prior
    return samples_ok[sel]

def logL_from_chain(chain, loglike, i, burnin=500):
    loglike_ok = loglike[burnin:,i]
    sel = np.isfinite(loglike_ok) # discard samples with logL = -inf which are outside of the prior
    return loglike_ok[sel]

# See eg Bolstad, Understanding computational Bayesian statistics
def gelman_rubin_MP_pick(chain, loglike, idx, param=0, burnin=500):
    chainData = []
    for i in idx:
        s = samples_from_chain(chain, loglike, i, param=param, burnin=burnin)
        if len(s) > 0:
            chainData.append(s)

    allData = np.concatenate(chainData)
    chainMeans = [np.mean(data) for data in chainData]
    chainVars = [np.var(data) for data in chainData]
    BoverN = np.var(chainMeans) # between chain variance B/n
    W = np.mean(chainVars)      # average of all the within-chain variances
    sigmaHat2 = W + BoverN
    m = len(chainData)
    VHat = sigmaHat2 + BoverN/m  # estimated variance
    try:
        R = VHat/W   # for values of m that are less than an adequate burn-in time, V should overestimate the variance of the target distribution, while W will underestimate the variance
    except:
        print "Error when computing Gelman-Rubin R statistic for"
        R = np.nan
    return R # Values of sqrt(R) that are less than 1.10 show acceptable convergence.

def find_good_chains(chain, loglike, param, Q_true, burnin=1500, Q_threshold=0.05, Q_extent_threshold=1000):
    '''
    Use bias as an indicator of convergence of chain
    Then compute Gelman-Rubin statistic for all good chains and return list of indices
    '''
    igood = []
    for i in range(len(chain)):
        s = samples_from_chain(chain, loglike, i, param=0, burnin=1500)
        if len(s) > 0:
            if (abs((Q_true - np.median(s))/Q_true) < Q_threshold) and (np.max(s) - np.min(s) < Q_extent_threshold):
#                 print i, np.median(s), np.max(s) - np.min(s)
                igood.append(i)

    return igood, gelman_rubin_MP_pick(chain, loglike, igood, burnin=1500)

def Compute_ACLs_Neff_for_chains(chain, loglike, param, burnin=1000):
    ACLs = []
    Neffs = []
    for i in range(len(chain)):
        s = samples_from_chain(chain, loglike, i, param, burnin=burnin)
        if len(s) > 0:
            # call slightly tweaked lalinference functions
            ACL = autocorrelation_length_estimate(s)
            Neff = effectiveSampleSize(s)[0]
        else:
            ACL = None
            Neff = None
        ACLs.append(ACL)
        Neffs.append(Neff)

    ACLs = np.array(ACLs)
    Neffs = np.array(Neffs)

    return ACLs, Neffs

def load_samples_safe(dataDir, SNR, burnin=1500, param=0, param_true=etafun(10.), Q_threshold=0.05, Q_extent_threshold=0.08, plot_me=False):
    """
    Load samples from double-spin emcee run. 
    Throw away samples before burnin and at -inf likelihood. 
    Keep only good chains and output Gelman-Rubin statistic.
    Convert to match.
    """
    chain = np.load(dataDir+'chain.npy')
    loglike = np.load(dataDir+'loglike.npy')

    # Analyse chains either for chirp mass or eta as these are the two most peaked distributions
    idx_good, R = find_good_chains(chain, loglike, param, param_true, burnin=burnin, Q_threshold=Q_threshold, Q_extent_threshold=Q_extent_threshold)
    ACLs, Neffs = Compute_ACLs_Neff_for_chains(chain, loglike, param, burnin=burnin)
    print 'Found %d good chains (Gelman-Rubin = %.2f) with a total of %d effective samples' % (len(idx_good), R, sum(Neffs[idx_good]))

    if plot_me:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(5,3.5))
        for i in idx_good:
            s = samples_from_chain(chain, loglike, i, param=0, burnin=burnin)
            ax1.hist(s, 50, histtype='step', normed=True, label=str(i)+': '+str(ACLs[i]));
            s = samples_from_chain(chain, loglike, i, param=1, burnin=burnin)
            ax2.hist(s, 50, histtype='step', normed=True, label=str(i)+': '+str(ACLs[i]));
        ax1.set_xlabel(r'$\eta$')
        ax2.set_xlabel(r'$M_c$')

    # Concatenate all good samples into one array for each parameter and the likelihood
    s_eta  = np.concatenate([samples_from_chain(chain, loglike, i, param=0, burnin=burnin) for i in idx_good])
    s_Mc   = np.concatenate([samples_from_chain(chain, loglike, i, param=1, burnin=burnin) for i in idx_good])
    s_chi1 = np.concatenate([samples_from_chain(chain, loglike, i, param=2, burnin=burnin) for i in idx_good])
    s_chi2 = np.concatenate([samples_from_chain(chain, loglike, i, param=3, burnin=burnin) for i in idx_good])
    s_logL = np.concatenate([logL_from_chain(chain, loglike, i, burnin=burnin) for i in idx_good])

    matchval = np.sqrt(2*s_logL)/SNR # convert to match

    s_q = qfun(s_eta)
    s_M = Mfun(s_Mc, s_eta)
    s_m1 = m1fun(s_Mc, s_q)
    s_m2 = m2fun(s_Mc, s_q)
    s_chieff = chi_eff(s_eta, s_chi1, s_chi2)
    s_chiPN = chi_PN(s_eta, s_chi1, s_chi2)

    return {'eta' : s_eta, 'Mc' : s_Mc, 'chieff' : s_chieff, 'chiPN' : s_chiPN,
            'chi1' : s_chi1, 'chi2' : s_chi2, 'Mtot' : s_M,
            'm1' : s_m1, 'm2' : s_m2, 'match' : matchval}
