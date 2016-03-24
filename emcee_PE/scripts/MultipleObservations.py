# 
__author__ = "Prayush Kumar <prayush.kumar@ligo.org>"

import os, sys, time
import commands as cmd
import numpy as np
from matplotlib import mlab, cm, use
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex' : True})
from mpl_toolkits.axes_grid1 import ImageGrid

from pydoc import help
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar, fmin
import scipy.integrate as si

from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate

try:
    sys.path.append( os.path.dirname(os.path.realpath(__file__)) + '/../src/' )
    sys.path.append( os.path.dirname(os.path.realpath(__file__)) )
except:
    sys.path.append( os.path.dirname(cmd.getoutput('pwd -P')) + '/../src/' )
    sys.path.append( os.path.dirname(cmd.getoutput('pwd -P')) )

sys.path.append( '/home/prayush/src/UseNRinDA/scripts/' )
sys.path.append( '/home/prayush/src/UseNRinDA/plotting' )

from utils import *
import pycbc.pnutils as pnutils
from PlotOverlapsNR import make_contour_plot_multrow
import h5py


# Constants
verbose  = True
vverbose = True
debug    = True

quantiles_68 = [0.16, 0.5, 0.84] # 1-sigma
quantiles_95 = [0.0228, 0.5, 0.9772] # 2-sigma ~ 95.4%
percentiles_68 = 100*np.array(quantiles_68)
percentiles_95 = 100*np.array(quantiles_95)
perc_int_95 = [2.28, 97.72]
perc_int_99_7 = [0.13, 99.87] # 3 sigma

CILevels=[90.0, 68.26895, 95.44997, 99.73002]

######################################################
# Function to make parameter bias plots
######################################################
linestyles = ['-', '--', '-.', '-x', '--o']
linecolors = ['r', 'g', 'b', 'k', 'm', 'y']
plotdir = 'plots/'
gmean = (5**0.5 + 1)/2.

# Figure settings
ppi=72.0
aspect=(5.**0.5 - 1) * 0.5
size=4.0 * 2# was 6
figsize=(size,aspect*size)
plt.rcParams.update({\
'legend.fontsize':11, \
'text.fontsize':11,\
'axes.labelsize':11,\
'font.family':'serif',\
'font.size':11,\
'xtick.labelsize':11,\
'ytick.labelsize':11,\
'figure.subplot.bottom':0.2,\
'figure.figsize':figsize, \
'savefig.dpi': 500.0, \
'figure.autolayout': True})





#######################################################
### FUNCTIONS 
#######################################################
def obtain_statistical_information_from_kde(\
                kde_func=None,\
                x_ref=0,\
                CILevel = 0.90,\
                xllimit=0, xulimit=4000,\
                verbose=False):
    '''
Given a p(x), calculate the median value of X and confidence level bounds.
Input:-

kde_func= Kernel Density Estimator function. should be p(X) = kde_func(X)
CILevel = Confidence level bound
xllimit = MIN Lower limit on all integrals over X
xulimit = MAX upper limit on all integrals over X

output:-
[lower confidence bound, median, upper confidence bound]
    '''
    eps = [(1. - CILevel)/2., 1. - (1. - CILevel)/2.]
    #
    denom = si.quad(kde_func, xllimit, xulimit, epsabs=1.e-16, epsrel=1.e-16)[0]
    if verbose:
        print >>sys.stdout, "Integral of PDF = %e" % denom
        sys.stdout.flush()
    def tmp_intfunc(L):
        if L < xllimit:
            return 0
        elif L > xulimit:
            L = xulimit
        return si.quad(kde_func, xllimit, L, epsabs=1.49e-12, epsrel=1.49e-12)[0] / denom
    #
    def tmp_rootfunc(L, eps): return np.abs(tmp_intfunc(L) - eps)
    def tmp_medianfunc(L): return np.abs(tmp_rootfunc(L, 0.5))
    def tmp_llimitfunc(L): return np.abs(tmp_rootfunc(L, eps[0]))
    def tmp_ulimitfunc(L): return np.abs(tmp_rootfunc(L, eps[-1]))
    #
    median = fmin(tmp_medianfunc, x_ref, xtol=1.e-16, ftol=1.e-16)
    llimit = fmin(tmp_llimitfunc, x_ref, xtol=1.e-16, ftol=1.e-16)
    ulimit = fmin(tmp_ulimitfunc, x_ref, xtol=1.e-16, ftol=1.e-16)
    #
    if False:
        median2 = minimize_scalar(tmp_medianfunc, bounds=(xllimit, xulimit), method='bounded', tol=1.e-16).x
        llimit2 = minimize_scalar(tmp_llimitfunc, bounds=(xllimit, xulimit), method='bounded', tol=1.e-16).x
        ulimit2 = minimize_scalar(tmp_ulimitfunc, bounds=(xllimit, xulimit), method='bounded', tol=1.e-16).x
    #
    if False and np.abs(median - median2)/median > 0.05:
        if verbose:
            print >>sys.stdout, "Root solvers don't match"
            print >>sys.stdout, [llimit, median, ulimit], [llimit2, median2, ulimit2]
            sys.stdout.flush()
        cond = llimit < median and median < ulimit and (ulimit - llimit) > 0.1*median
        cond2 = llimit2 < median2 and median2 < ulimit2 and (ulimit2 - llimit2) > 0.1*median2
        if cond2: return [llimit2, median2, ulimit2]
    #
    return [llimit, median, ulimit]

def sample_snr_from_volume(SNRvec):
    '''
Sample the SNR of signals, as if they are distributed uniformly in
the observable volume, i.e. 4/3 pi D_Luminosity^3
    '''
    SNRvec    = np.array(SNRvec)
    distvec   = 1./SNRvec
    distprob = distvec**2
    #total_prob = sum(distprob)
    # Now generate the SNR value, in proportion to 'uniform distribution of sources in volume'
    rnd_tmp = np.random.uniform(0, sum(distprob))
    cumsum = 0
    for i in range(len(SNRvec)):
        cumsum += distprob[i]
        if cumsum > rnd_tmp: return SNRvec[i]
    #
    raise RuntimeError("SNR sampling not being done properly")
        

#######################################################
### CALCULATION CLASSES
#######################################################
class multiple_observation_results:
    '''
######################################################
Inputs:-

N                : number of events to be generated
NSLambda                   : Lambda of the NS in the generated events
source_distribution        : How are the sources distributed in the
                                universe? Allowed: UniformInVolume
only_positive_aligned_spins: Restrict samples to positive-aligned
                                BH spins
kernel           : Kernel function. Allowed options:-
    - "biw" for biweight
    - "cos" for cosine
    - "epa" for Epanechnikov
    - "gau" for Gaussian.
    - "tri" for triangular
    - "triw" for triweight
    - "uni" for uniform
bw_method        : Bandwidth prescription. KDE is very SENSITIVE to it
                    Allowed options:-
    - "scott" - 1.059 * A * nobs ** (-1/5.), where A is
      `min(std(X),IQR/1.34)`
    - "silverman" - .9 * A * nobs ** (-1/5.), where A is
      `min(std(X),IQR/1.34)`
    - "normal_reference" - C * A * nobs ** (-1/5.), where C is
      calculated from the kernel. Equivalent (up to 2 dp) to the
      "scott" bandwidth for gaussian kernels. See bandwidths.py
    - If a float is given, it is the bandwidth.


Needs defined:
chi2vec : List of BH spins available
qvec    : List of mass-ratios available
SNRvec  : List of SNR's available


What it does:
 Samples systems uniformly in
  1) BH mass (from available options)
  2) BH spin
  3) source volume distribution

######################################################

    '''
    def __init__(self, lambda_posterior_chains,\
                 N = 10,\
                 NSLambda = 1000,\
                 chi2vec=None,\
                 qvec=None,\
                 SNRvec=None,\
                 source_distribution='UniformInVolume',\
                 only_positive_aligned_spins=True,\
                 kernel='gau',\
                 bw_method='scott',\
                 verbose=False):
        ### REMOVE BH spins < 0
        if verbose:
            print >>sys.stdout, "Removing negative BH spins"
            sys.stdout.flush()
        if only_positive_aligned_spins:
            chi2vec = np.array(chi2vec)
            chi2vec = chi2vec[chi2vec >= 0]
        #
        self.lambda_posterior_chains = lambda_posterior_chains
        #
        self.N       = N
        self.NSLambda= NSLambda
        self.chi2vec = chi2vec
        self.qvec    = qvec
        self.SNRvec  = SNRvec
        #
        self.source_distribution = source_distribution
        self.only_positive_aligned_spins = only_positive_aligned_spins
        self.kernel = kernel
        self.bw_method = bw_method
        #
        self.verbose = verbose
        #
        self.print_info()
        return
    ###
    def generate_events(self, lambda_posterior_chains=None,\
                        NSLambda=None,\
                        qvec=None, chi2vec=None, SNRvec=None):
        '''
        Returns is a list of events. For each event, two objects are returned:
        1. array of posterior samples
        2. fitted kernel density estimator's evaluate method
        '''
        ####
        N = self.N
        if lambda_posterior_chains == None:
            lambda_posterior_chains = self.lambda_posterior_chains
        if SNRvec == None: SNRvec = self.SNRvec
        if chi2vec == None: chi2vec = self.chi2vec
        if qvec == None: qvec = self.qvec
        if NSLambda == None: NSLambda = self.NSLambda
        # Initiate primary data structure
        chain_set = []
        ###
        ## Loop over as many times as many events are to be generated
        ###
        if self.verbose:
            print >>sys.stdout, "Creating Events >>"
            sys.stdout.flush()
        for i in range(N):
            #
            if self.verbose:
                print >>sys.stdout, "  Event %d" % i
                sys.stdout.flush()
            # DRAW TWO uniformly distributed random samples for (q, chiBH)
            rnd_q = qvec[np.int(np.random.uniform(0, len(qvec)))]
            rnd_chiBH = chi2vec[np.int(np.random.uniform(0, len(chi2vec)))]
            #
            # Sample the SNR depending on source distribution specified
            if 'UniformInVolume' in self.source_distribution:
                rnd_SNR = sample_snr_from_volume(SNRvec)
            else:
                raise IOError("Only UniformInVolume is supported as a source distribution")
            #
            if self.verbose:
                print >>sys.stdout, "Sampled q = %f, chiBH = %f, SNR = %f" % (rnd_q, rnd_chiBH, rnd_SNR)
                sys.stdout.flush()
            ####
            ## Store the primary (RAW) data
            lambda_chain = lambda_posterior_chains[rnd_q][rnd_chiBH][NSLambda][rnd_SNR]
            ####
            ## Now create the Kernel Density Estimator (KDE)
            #
            lambda_kde = KDEUnivariate(lambda_chain)
            try:
                lambda_kde.fit(kernel=self.kernel, bw=self.bw_method, fft=True)
            except:
                lambda_kde.fit(kernel=self.kernel, bw=self.bw_method, fft=False)
            #
            ####
            ## Normalize the chain's integral
            xllimit, xulimit = [0, 4000]
            lambda_kde_norm = si.quad(lambda_kde.evaluate, xllimit, xulimit,\
                                        epsabs=1.e-12, epsrel=1.e-12)[0]
            #lambda_gaussian_kde = gaussian_kde(lambda_chain)
            #lambda_multivariate_kde = KDEMultivariate(lambda_chain, bw=0.2*np.ones_like(lambda_chain), var_type='c')
            #lambda_skl_kde = KernelDensity(bandwidth=0.2)
            #lambda_skl_kde.fit(lambda_chain[:, np.newaxis])    
            ####
            ## Add the RAW DATA and KDE to the PRIMARY DATA STRUCTURE
            ##
            chain_set.append( [lambda_chain, lambda_kde.evaluate, lambda_kde_norm] )
        #
        self.chain_set = chain_set
        self.FULL_chain_set = self.chain_set
        self.generate_cumulative_normalizations() # Normalizations
        self.print_info()
        return chain_set
    #####
    ### Return the num-th of all probability distribution functions
    ###   i.e. p(num, X), where num is the index of the chain, evaluated
    ###   at X =  L.
    def chain_kde(self, L, num=-1, normed=False):
        '''
        Get the product of all posterior distributions, for a given lambda value
        i.e. returns "p(L) = p1(L) x p2(L) x p3(L) x p4(L) x .. x pN(L)"
        
        Note: num goes from 1 onwards
        
        '''
        if num < 0: num = self.N
        _, kde_func, kde_norm = self.FULL_chain_set[num-1]
        kde_val = kde_func(L)
        if normed: kde_val = kde_val / kde_norm
        return kde_val
    #####
    ### Returns P(num, X) = \PI_0^{num-1} p(i, X)
    ### i.e. the PRODUCT of first num probability density functions, 
    ### evaluated at X = L.
    ### Takes in a LIST of indices too!
    ###
    def chain_kde_product(self, L, num=-1, normed=False):
        '''
        Get the product of all posterior distributions, for a given lambda value
        i.e. returns "p(L) = p1(L) x p2(L) x p3(L) x p4(L) x .. x pN(L)"
        '''
        prod = 1.0
        if type(num) == list:
            for i in num:
                prod = prod * self.chain_kde(L, i, normed=normed)
        else:
          if num < 0: num = self.N
          for i in range(num):
              prod = prod * self.chain_kde(L, i+1, normed=normed)
        #
        return prod
    #####
    ### Calculate the NORMALIZATION INTEGRAL for any function P(x),
    ###  s.t. P(X) = \PI_0^{num-1} p(x)
    ### i.e. product of first 'num' p(X) densities
    def chain_kde_product_norm(self, num=-1, xllimit=0, xulimit=4000):
        '''
        Calls chain_kde_product and returns its normalization integral
        '''
        def wrap_product(L): return self.chain_kde_product(L, num=num)
        return si.quad(wrap_product, xllimit, xulimit, epsabs=1.49e-16)[0]    
    #
    ###
    def generate_cumulative_normalizations(self):
        if not hasattr(self, 'normalization_data'): self.normalization_data = {}
        else: return
        #
        for i in range(self.N):
            self.normalization_data[i] = self.chain_kde_product_norm(num=range(1, i+1))
            if self.verbose:
              print " .. NORM over first %d events = %e" % (i+1, self.normalization_data[i])            
        return
    #####
    ### Calculate various statistics from the probability density 
    ### function P(X), where P(X) = \PI_0^{num-1} p(i, x)
    ###
    ###
    def statistics_from_events(self, num=-1):
        '''
This function computes the median and confidence level bounds on a given set
of events. the events can be specified as an integer value of num of a list of
integers for num. 

n is integer: first 'n' chains are used to get statistics
n is a list of integers: all the indexed chains are used
        '''
        itime = time.time()
        if self.verbose:
          print "Here we compute statistics from the join distribution for the event set provided by user."
        #####
        ### Calculate the Normalized (in the integration sense) KDE
        def tmp_chain_kde_product(L): 
            return self.chain_kde_product(L, num=num, normed=False)
        #
        #####
        ### Calculate Median, Standard Deviation, Confidence intervals from the COMBINED 
        # probability distribution
        ll, ml, ul = obtain_statistical_information_from_kde(\
                            kde_func=tmp_chain_kde_product,\
                            x_ref=self.NSLambda, verbose=self.verbose)
        fractional_err = np.abs(ul - ll)/self.NSLambda
        if self.verbose: print "All done in %f seconds!" % (time.time() - itime)
        return [ml, fractional_err, ll, ul]
    #
    ###
    def generate_cumulative_statistics(self):
        '''
Gnerate statistics for all of stored events, cumulatively including more & more
        '''
        itime = time.time()
        if not hasattr(self, 'normalization_data'):
          self.generate_cumulative_normalizations()
        if not hasattr(self, 'statistical_data'):
          self.statistical_data = {}
        if self.verbose:
            print >>sys.stdout, "\n\nComputing Statistics now >>"
            sys.stdout.flush()
        ####
        for i in range(self.N):
            if self.verbose:
              print " >> For the first %d events! >>>" % (i+1)
            self.statistical_data[i] =\
                                  self.statistics_from_events(num=range(1, i+1))
        #####
        if self.verbose:
          print "All done in %f seconds!" % (time.time() - itime)
          self.print_info()
        return self.statistical_data
    #
    def print_info(self):
        print """
For each event:-
self.chain_sets      = posterior samples, posterior KDE are stored
self.FULL_chain_sets = same as above (read-only)
self.N                      : No of events
self.qvec/chi2vec/Lambdavec : parameters array
self.normalization_data     : Norm of \PI_I p(I,X)
self.statistical_data       : [CL lower limit, Median, CL upper limit] for 
                               cumulative number of events
Functions:

self.chain_kde              = get p(Lambda) for chain number 'num' (STARTING FROM 1)
self.chain_kde_product      = get p(Lambda) for the product of first 'num' posterior
                              distributions. num is either chain numbers' list or
                              their max
self.kde_product_norm       = get the \int chain_kde_product() dLambda for a given 
                              chain set, specified by the set of indices 'num'
self.generate_cumulative_normalizations = populate 
self.statistics_from_events = get a chain set and get statistics about it as one entity
self.generate_cumulative_statistics = call "statistics_from_events" for first 1, first 2,
                                       .. first N independently
            
            """
#
