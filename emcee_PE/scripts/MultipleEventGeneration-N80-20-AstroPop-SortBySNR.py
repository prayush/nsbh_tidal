#!/usr/bin/env python
# Tools for analysis of MCMC samples from emcee for spin estimation project
# Copyright 2015
# 
__author__ = "Prayush Kumar <prayush.kumar@ligo.org>"

import os, sys, time
import commands as cmd
import numpy as np
from matplotlib import mlab, cm, use
use('Agg')
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex' : True})
from mpl_toolkits.axes_grid1 import ImageGrid


from pydoc import help
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
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
sys.path.append( '/home/prayush/src/nsbh_tidal/emcee_PE/scripts/' )
sys.path.append( '/home/prayush/src/nsbh_tidal/emcee_PE/src/' )

from utils import *
import pycbc.pnutils as pnutils
from PlotOverlapsNR import make_contour_plot_multrow
import h5py

import MultipleObservations as MO
#reload(MO)

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
gmean = (5**0.5 + 1)/2.

# Figure settings
ppi=72.0
aspect=(5.**0.5 - 1) * 0.5
size=4.0 * 2# was 6
figsize=(size,aspect*size)
plt.rcParams.update({    'legend.fontsize':16,     'text.fontsize':16,    'axes.labelsize':16,    'font.family':'serif',    'font.size':16,    'xtick.labelsize':16,    'ytick.labelsize':16,    'figure.subplot.bottom':0.2,    'figure.figsize':figsize,     'savefig.dpi': 500.0,     'figure.autolayout': True})

# hist(PercentileInterval(s1[0], pc=99.0), 100);


# In[2]:

######################################################
# Set up parameters of templates
######################################################

inject_tidal  = True
recover_tidal = True

chi1 = 0.   # small BH
chi2vec = [-0.5, 0, 0.5, 0.74999]  # larger BH
mNS = 1.35
qvec = [2, 3, 4, 5]
#Lambdavec = [0]#[500, 800, 1000]
if inject_tidal:
  SNRvec = [10, 15, 20, 30, 50, 70]
  Lambdavec = [500, 800, 1000, 1500, 2000]
  #Lambdavec = [1500, 2000]
else:
  raise RuntimeError("This script combines Tidal observations")

######################################################
# PRIORs
######################################################

# Taking an uninformed prior
m1min = 1.2
m1max = 15.
m2min = 1.2 
m2max = 25

LambdaMax = 4000
Lambdastdev = 100

######################################################
# Set up RUN parameters
######################################################
if inject_tidal: sigstring = 'T'
else: sigstring = 'N'
if recover_tidal: tmpstring = 'T'
else: tmpstring = 'N'
simstring = sigstring + tmpstring + '_'

Nwalkers = [100]
Nsamples = [150000]
Nburnin  = 2000


# ## READ IN DATA

# In[3]:

######################################################
# Read in parameter biases and other data
######################################################
datafile_postfix = 'TT_ParameterBiasesAndConfidenceIntervals.h5'
datafile_dir     = '/home/prayush/projects/nsbh/TidalParameterEstimation/nsbh_tidal/PEData/'

# Load in all the Posterior distribution function samples
all_posterior_chains   = {}
lambda_posterior_chains= {}
all_loglike_chains     = {}
all_chains_summary     = {}
all_chains_GelmanRubin = {}

data_dicts = [all_posterior_chains,              lambda_posterior_chains,              all_loglike_chains,              all_chains_summary,              all_chains_GelmanRubin]

for q in qvec:
  datafile = os.path.join(datafile_dir, ('q%1d_' % q) + datafile_postfix)
  fin = h5py.File(datafile, 'r')
  for tmp_dict in data_dicts: tmp_dict[q] = {}
  #
  for chiBH in chi2vec:
    for tmp_dict in data_dicts: tmp_dict[q][chiBH] = {}
    #
    for Lambda in Lambdavec:
      for tmp_dict in data_dicts: tmp_dict[q][chiBH][Lambda] = {}
      #
      for SNR in SNRvec:
        for tmp_dict in data_dicts: tmp_dict[q][chiBH][Lambda][SNR] = {}        
        l1grp, l2grp, l3grp, l4grp =                   ['q%.1f.dir' % q, 'chiBH%.2f.dir' % chiBH,                   'LambdaNS%.1f.dir' % Lambda, 'SNR%.1f.dir' % SNR]
        #
        l5dset = 'chain.dat'
        all_posterior_chains[q][chiBH][Lambda][SNR] = fin[l1grp][l2grp][l3grp][l4grp][l5dset].value
        lambda_posterior_chains[q][chiBH][Lambda][SNR] = fin[l1grp][l2grp][l3grp][l4grp][l5dset].value[:,-1]
        #
        l5dset = 'loglikelihood.dat'
        all_loglike_chains[q][chiBH][Lambda][SNR] = fin[l1grp][l2grp][l3grp][l4grp][l5dset].value
        #
        l5dset = 'summary.dat'
        all_chains_summary[q][chiBH][Lambda][SNR] = fin[l1grp][l2grp][l3grp][l4grp][l5dset].value
        #
        l5dset = 'GelmanRubin.dat'
        all_chains_GelmanRubin[q][chiBH][Lambda][SNR] = fin[l1grp][l2grp][l3grp][l4grp][l5dset].value        
  #
  fin.close()


# ## READ/ORGANIZE DATA

# In[4]:

## INITIATE A SET OF MULTIPLE OBSERVATIONS

# Constants
NEVENTS = int(sys.argv[1])
#
RAND = np.zeros(len(Lambdavec))
if len(sys.argv) > 2:
  RAND = np.int64(sys.argv[2].split('_'))
#
INDEX   = 20
print "RAND = ", RAND

# In[5]:

xx = {}
for i, NSL in enumerate(Lambdavec):
    print "Initiating event class with Lambda = %f" % NSL
    xx[NSL] = MO.multiple_observation_results(lambda_posterior_chains,\
                            N=NEVENTS, NSLambda=NSL,\
                            chi2vec=chi2vec, qvec=qvec,\
                            SNRvec=SNRvec,\
                            source_distribution='UniformIn3DVolumeInclination',\
                            kernel='gau',\
                            RND=RAND[i],\
                            output_tag='SNRSorted_',\
                            verbose=True)


# In[6]:

plotdir = 'plots/N%d-%d' % (NEVENTS, INDEX)

for NSL in Lambdavec:
    plotdir = plotdir + ('_%d' % xx[NSL].RND)

print "plot directory is ", plotdir

try: os.makedirs(plotdir)
except: pass


# In[7]:

fout = open(os.path.join(plotdir, 'chain_indices.dat'), 'w')
#
for NSL in Lambdavec:
    fout.write('%d\t%d\n' % (NSL, xx[NSL].RND))
#
fout.close()


# In[8]:

## LOAD A SEQUENCE OF EVENTS and SORT EVENTS ACCORDING TO SNR !!
for i, NSL in enumerate(Lambdavec):
    print "---- For Lambda = %f" % NSL
    xx[NSL].load_events(RAND[i], sort_events_column=2) # cols are M, eta, SNR, L



# Generate ALL Statistics with the population
for NSL in Lambdavec:
    print "---- For Lambda = %f" % NSL
    xx[NSL].generate_cumulative_statistics()


# # PLOT the DATA

# ## SINGLE LAMBDA values

# In[10]:

######################################################
# Function to make parameter bias plots
######################################################
linestyles = ['-', '--', '-.', '-x', '--o']
linecolors = ['r', 'g', 'b', 'k', 'm', 'y']
gmean = (5**0.5 + 1)/2.

# Figure settings
ppi=72.0
aspect=(5.**0.5 - 1) * 0.5
size=4.0 * 2# was 6
figsize=(size,aspect*size)
plt.rcParams.update({    'legend.fontsize':16,     'text.fontsize':16,    'axes.labelsize':16,    'font.family':'serif',    'font.size':16,    'xtick.labelsize':16,    'ytick.labelsize':16,    'figure.subplot.bottom':0.2,    'figure.figsize':figsize,     'savefig.dpi': 500.0,     'figure.autolayout': True})


# In[11]:

# Cumulative histograms of probability distribution functions
# as a function of number of events N
lvalues = np.linspace(50, 4950, 250)

for lambda_val in Lambdavec:
    xx2 = xx[lambda_val]
    xx2.chain_set = xx2.FULL_chain_set
    #
    fig = figure()
    ax = fig.add_subplot(111)
    all_xys = []
    try:
        print >> sys.stdout, "Inside try clause for lambda = ", lambda_val
        sys.stdout.flush()
        for i in range(NEVENTS, 0, -1):
            if False:
                print >>sys.stdout, "Inside for loop, i = ", i
                sys.stdout.flush()
            #
            x1, y1 = lvalues, xx2.chain_kde_product(lvalues, num=i)/xx2.chain_kde_product_norm(num=i)
            all_xys.append([x1, y1])
            #    
            plot(x1, y1, lw=1.5, label='$N=%d$' % (i+0))
            grid(True)
            hold(True)
            axvline(x=lambda_val, color='k', lw=2, linestyle='--')
    except RuntimeError as herr:
        print "PKError was:-\n", herr
        xx2.chain_set = FULL_chain_set
    #
    legend(loc='best', ncol=2)
    #xlim(500,1400)
    title('$\Lambda_\mathrm{NS}=%.1f$' % lambda_val)
    xlabel('$\Lambda_\mathrm{NS}$')
    ylabel('Probability density')
    #
    savefig(os.path.join(plotdir, 'pdfLambda_vs_N_L%d_noCI.pdf' % int(lambda_val)))

#plot(lvalues/xx2.NSLambda, xx2.chain_kde_product(lvalues)/xx2.kde_product_norm(), 'k', lw=2)


# In[12]:

print plotdir, xx.keys()
#!ls plots/N50-5/


# In[13]:

# Cumulative histograms of probability distribution functions
# with 90% confidence intervals, as a function of number of events N
lvalues = np.linspace(50, 4950, 250)

for lambda_val in Lambdavec:
    xx2 = xx[lambda_val]
    xx2.chain_set = xx2.FULL_chain_set
    #
    figure()
    all_xys = []
    try:
        print >> sys.stdout, "Inside try clause for lambda = ", lambda_val
        sys.stdout.flush()
        for i in range(NEVENTS, 0, -1):
            if False:
                print >>sys.stdout, "Inside for loop, i = ", i
                sys.stdout.flush()
            #
            x1, y1 = lvalues, xx2.chain_kde_product(lvalues, num=i)/xx2.chain_kde_product_norm(num=i)
            all_xys.append([x1, y1])
            #
            median_val = xx2.statistical_data[i-1][0]
            ulimit_val = xx2.statistical_data[i-1][2]
            llimit_val = xx2.statistical_data[i-1][3]
            #
            plot(x1, y1, lw=1.5, label='$N=%d$' % (i+0))
            grid(True)
            hold(True)
            axvline(x=llimit_val, color='k', lw=0.7, alpha=1.0 * i/NEVENTS)
            axvline(x=ulimit_val, color='k', lw=0.7, alpha=1.0 * i/NEVENTS)
            #"
            axvline(x=lambda_val, color='k', lw=2, linestyle='--')
    except RuntimeError as herr:
        print "PKError was:-\n", herr
        xx2.chain_set = FULL_chain_set
    #
    legend(loc='best', ncol=3, fontsize=8)
    #xlim(500,1400)
    title('$\Lambda_\mathrm{NS}=%.1f$' % lambda_val)
    xlabel('$\Lambda_\mathrm{NS}$')
    ylabel('Probability density')
    #
    savefig(os.path.join(plotdir, 'pdfLambda_vs_N_L%d.pdf' % int(lambda_val)))

#plot(lvalues/xx2.NSLambda, xx2.chain_kde_product(lvalues)/xx2.kde_product_norm(), 'k', lw=2)


# In[14]:

# Simple errorbar plots, one for each NS lambda, showing the median 
# and 90% confidence intervals, as a function of number of events N
linecolor = ['k', 'b', 'r', 'k', 'g', 'm', 'brown', 'orange']
Narray = np.arange(1, NEVENTS+1)

for idx, NSL in enumerate(Lambdavec):
    print "---- For Lambda = %f" % NSL
    xx2 = xx[NSL]
    figure()
    median_array = [xx2.statistical_data[i][0] for i in range(NEVENTS)]
    errorbar_array = [[xx2.statistical_data[i][0] - xx2.statistical_data[i][2],                   xx2.statistical_data[i][3] - xx2.statistical_data[i][0]]  for i in range(NEVENTS)]
    #
    errorbar(Narray, median_array, yerr=np.transpose(errorbar_array),                 fmt=linecolor[idx]+'o', lw=1, label='$\Lambda=%.1f$' % NSL)
    hold(True)
    axhline(y=NSL, color=linecolor[idx], lw=2, linestyle='--')
    hold(True)
    #
    grid(True)
    legend(loc='best')
    xlim(0.9, NEVENTS + .1)
    #ylim(500,1800)
    xlabel('Number of Events')
    ylabel('Neutron Star Compactness')
    #
    savefig(os.path.join(plotdir, 'ErrorBarsLambda_vs_N_L%d.pdf' % int(NSL)))


# In[ ]:




# In[15]:

# Simple filled-region plots, one for each NS lambda, showing the median 
# and 90% confidence intervals, as a function of number of events N
linecolor = ['b', 'r', 'k', 'g', 'm', 'brown', 'orange']
Narray = np.arange(1, NEVENTS+1)

for idx, NSL in enumerate(Lambdavec):
    print "---- For Lambda = %f" % NSL
    xx2 = xx[NSL]
    figure()
    median_array = [xx2.statistical_data[i][0] for i in range(NEVENTS)]
    llimit_array = [xx2.statistical_data[i][2] for i in range(NEVENTS)]
    ulimit_array = [xx2.statistical_data[i][3] for i in range(NEVENTS)]
    #
    fill_between(Narray, llimit_array, y2=ulimit_array,                 color=linecolor[idx], alpha=0.5, label='$\Lambda=%.1f$' % NSL)
    hold(True)
    plot(Narray, median_array, linecolor[idx]+'-o', alpha=0.8 )
    hold(True)
    axhline(y=NSL, color=linecolor[idx], lw=2, linestyle='--')
    hold(True)
    #
    grid(True)
    legend(loc='best')
    xlim(0.9, NEVENTS + .1)
    #ylim(500,1800)
    xlabel('Number of Events')
    ylabel('Neutron Star Compactness')
    #
    savefig(os.path.join(plotdir, 'FillBetweenErrorBarsLambda_vs_N_L%d.pdf' % int(NSL)))


# In[ ]:




# In[16]:

# Filled-region plots, one for each NS lambda, showing the median 
# and 90% confidence intervals, as a function of number of events N
# Shown are two independent regions. One in pale red shows the recovered
# confidence interval for neutron star compactness (raw value), as
# we accumulate more events. The y-values are shown on the *right*
# y-axis. The red dashed line shows the true value 
# of Lambda, which this region plot must tend towards. The dotted lines
# are contours of 1/sqrt(N), drawn to aid the eye.
# The second (blue) region shows the recovered confidence interval
# for lambda, normalized by its true value, as a function of the 
# number of observed events N. The recovered median is shown by the
# line-circled curve. The pair of horizontal green dashed
# lines show the \pm 25\% and \pm 50\% symmetric error bounds around
# the true value.
linecolor = ['c', 'grey', 'r', 'k', 'g', 'm', 'brown', 'orange']
Narray = np.arange(1, NEVENTS+1)

for idx, NSL in enumerate(Lambdavec):
    print "---- For Lambda = %f" % NSL
    fig = figure()
    ax = fig.add_subplot(111)
    xx2 = xx[NSL]
    median_array = [xx2.statistical_data[i][0] for i in range(NEVENTS)]
    llimit_array = [xx2.statistical_data[i][2] for i in range(NEVENTS)]
    ulimit_array = [xx2.statistical_data[i][3] for i in range(NEVENTS)]
    #
    norm_fac = xx2.NSLambda
    ##
    # RELATIVE ERROR REGION
    ax.fill_between(Narray, np.array(llimit_array)/norm_fac,                 y2=np.array(ulimit_array)/norm_fac,                 color=linecolor[0], alpha=0.75, label='$\Lambda=%.1f$' % NSL)
    hold(True)
    ax.plot(Narray, np.array(median_array)/norm_fac, linecolor[0]+'-o', alpha=0.8 )
    hold(True)
    #
    ax.axhline(y=NSL/norm_fac, color='g', lw=2, linestyle='--')
    ax.axhline(y=xx2.NSLambda/norm_fac-0.25, color='g', lw=1, alpha=1, linestyle='dotted')
    ax.axhline(y=xx2.NSLambda/norm_fac+0.25, color='g', lw=1, alpha=1, linestyle='dotted')
    ax.axhline(y=xx2.NSLambda/norm_fac-0.5, color='g', lw=1, alpha=1, linestyle='--')
    ax.axhline(y=xx2.NSLambda/norm_fac+0.5, color='g', lw=1, alpha=1, linestyle='--')
    fig.hold(True)
    ax.text(10, 6, "$y\propto 1/\sqrt{x}$", alpha=0.65)

    ax2 = ax.twinx()
    ##
    # ABSOLUTE ERROR REGION
    ax2.fill_between(Narray, np.array(llimit_array),                 y2=np.array(ulimit_array),                 color=linecolor[1], alpha=0.2, label='$\Lambda=%.1f$' % NSL)
    ax2.axhline(y=NSL, color=linecolor[1] , lw=2, linestyle='--')
    
    ax2.set_label('Neutron Star Compactness')
    
    hold(True)
    #
    ylow, yhigh = ax.get_ylim()
    ylow2, yhigh2 = ax2.get_ylim()
    scale_fac = yhigh2 / yhigh
    for y_val in np.linspace(ylow + 0.7, yhigh+0.7, num=7):
        #ax.plot(np.arange(1, NEVENTS+1, 1), 1. + y_val / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
        #    'k', linestyle='dotted', alpha=1, lw=2)
        ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),            'grey', linestyle='dotted', alpha=0.6, lw=1)
    #
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)
    ax2.yaxis.grid(False)
    ax2.xaxis.grid(True)
    
    #legend(loc='best')
    title('$\Lambda=%.1f$' % NSL)
    ax2.set_xlim(0.9, NEVENTS + .1)
    ax.set_ylim(0,8)
    ax2.set_ylim(0, 4000)
    ax.set_xlabel('Number of Events')
    ax.set_ylabel('Neutron Star Compactness$\,/\Lambda_\mathrm{true}$')
    ax2.set_ylabel('Neutron Star Compactness')
    #
    savefig(os.path.join(plotdir, 'FillBetweenNormErrorBarsLambda_vs_N_L%d.pdf' % int(NSL)))


# ## MULTIPLE LAMBDA values

# In[ ]:




# In[17]:

# Shown is the quantity (measured - true)/true lambda in line-circled
# curves. The measured value here is the median of the probability distribution
# obtained by accumulating information from N events. About the medians
# are shown actual 90% confidence intervals on the "measured" value,
# normalized to be on the same y-axis as the median. For e.g., take
# the case when the neutron star has lambda = 2000. After 10 events,
# the yellow errorbars range in [-0.25, 0.7], and the median (filled
# yellow circle) is close to 0.125. Therefore, we measured  value is
# 2000 * (1 + 0.125) + (0.7 * 2000) - (0.25 * 2000) = 2250^{+700}_{-500}.
# Grey dotted curves are contours of 1/sqrt(N). Horizontal dashed green lines
# demarkate +-25% and +-50% errors.

linecolor = ['k','r', 'g', 'b', 'y', 'r', 'm', 'k', 'orange']
Narray = np.arange(1, NEVENTS+1)

fig = figure()
ax = fig.add_subplot(111)
for idx, NSL in enumerate(Lambdavec):
    print "---- For Lambda = %f" % NSL
    xx2 = xx[NSL]
    median_array = [xx2.statistical_data[i][0] for i in range(NEVENTS)]
    errorbar_array = [[xx2.statistical_data[i][0] - xx2.statistical_data[i][2],                   xx2.statistical_data[i][3] - xx2.statistical_data[i][0]]  for i in range(NEVENTS)]
    #
    median_array = (np.array(median_array) - NSL) / NSL
    errorbar_array = np.array(errorbar_array) / NSL
    #
    errorbar(Narray, median_array, yerr=np.transpose(errorbar_array),                 fmt=linecolor[idx]+'-o', lw=1+1.2*idx, label='$\Lambda=%.1f$' % NSL, alpha=0.8-0.*idx)
    hold(True)
    Narray = Narray + 0.1
#
axhline(y=xx2.NSLambda/NSL -1, color='k', lw=1, alpha=1, linestyle='dashed')
axhline(y=xx2.NSLambda/NSL-0.5-1, color='m', lw=1, alpha=1, linestyle='--')
axhline(y=xx2.NSLambda/NSL+0.5-1, color='m', lw=1, alpha=1, linestyle='--')
axhline(y=xx2.NSLambda/NSL-0.25-1, color='m', lw=1, alpha=1, linestyle='dashed')
axhline(y=xx2.NSLambda/NSL+0.25-1, color='m', lw=1, alpha=1, linestyle='dashed')

ylow, yhigh = ax.get_ylim()
for y_val in np.linspace(ylow + 0.7, yhigh+0.7, num=15):
    ax.plot(np.arange(1, NEVENTS+1, 1), 0+ y_val / np.sqrt(np.arange(1, NEVENTS+1, 1)),        'k', linestyle='dotted', alpha=0.7, lw=1)
    #ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
    #    'k', linestyle='dotted', alpha=0.9, lw=1.2)
    
#
#ax.set_yscale('log')
grid(True, which='both')
ax.yaxis.grid(False)
legend(loc='best')
xlim(0.9, NEVENTS + .8)
ylim(-1, 1)
xlabel('Number of Events')
ylabel('Measured Neutron Star\n Compactness $/\Lambda_\mathrm{true}\, -1$')
savefig(os.path.join(plotdir, 'RelErrorBarsLambda_vs_NShifted_AllLambda.pdf'))


# In[18]:

# [Same as the previous one, but with the y axis on a log scale]
# Shown is the quantity (measured - true)/true lambda in line-circled
# curves. The measured value here is the median of the probability distribution
# obtained by accumulating information from N events. About the medians
# are shown actual 90% confidence intervals on the "measured" value,
# normalized to be on the same y-axis as the median. For e.g., take
# the case when the neutron star has lambda = 2000. After 10 events,
# the yellow errorbars range in [-0.25, 0.7], and the median (filled
# yellow circle) is close to 0.125. Therefore, we measured  value is
# 2000 * (1 + 0.125) + (0.7 * 2000) - (0.25 * 2000) = 2250^{+700}_{-500}.
# Grey dotted curves are contours of 1/sqrt(N). Horizontal dashed green lines
# demarkate +-25% and +-50% errors.
linecolor = ['k', 'b', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['k', 'b', 'g', 'r', 'c', 'k', 'orange']
linecolor = ['k', 'b', 'g', 'r', 'y', 'k', 'orange']

Narray = np.arange(1, NEVENTS+1)

fig = figure()
ax = fig.add_subplot(111)
for idx, NSL in enumerate(Lambdavec):
    print "---- For Lambda = %f" % NSL
    xx2 = xx[NSL]
    median_array = [xx2.statistical_data[i][0] for i in range(NEVENTS)]
    errorbar_array = [[xx2.statistical_data[i][0] - xx2.statistical_data[i][2],                   xx2.statistical_data[i][3] - xx2.statistical_data[i][0]]  for i in range(NEVENTS)]
    #
    median_array = np.array(median_array) / NSL
    errorbar_array = np.array(errorbar_array) / NSL
    #
    errorbar(Narray, median_array, yerr=np.transpose(errorbar_array),                 fmt=linecolor[idx]+'-o', lw=2+ 0.8*(idx+0), label='$\Lambda=%.1f$' % NSL, alpha=0.8-0.03*idx)
    hold(True)
    Narray = Narray + 0.1
#
axhline(y=xx2.NSLambda/NSL, color='k', lw=1, alpha=0.5, linestyle='dashed')
axhline(y=xx2.NSLambda/NSL-0.5-1, color='m', lw=1, alpha=1, linestyle='--')
axhline(y=xx2.NSLambda/NSL+0.5-1, color='m', lw=1, alpha=1, linestyle='--')
axhline(y=xx2.NSLambda/NSL-0.25-1, color='m', lw=1, alpha=1, linestyle='dashed')
axhline(y=xx2.NSLambda/NSL+0.25-1, color='m', lw=1, alpha=1, linestyle='dashed')

#
ax.set_yscale('log')
ylow, yhigh = ax.get_ylim()
for y_val in np.linspace(ylow + 0.7, yhigh+0.7, num=15):
    ax.plot(np.arange(1, NEVENTS+1, 1), 1 + y_val / np.sqrt(np.arange(1, NEVENTS+1, 1)),        'k', linestyle='dotted', lw=1, alpha=0.4)
    #ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
    #    'k', linestyle='dotted', alpha=0.9, lw=1.2)

#
#ax.set_yscale('log')
grid(False, which='both')
ax.yaxis.grid(False)
ax.xaxis.grid(True)


xlim(0.9, NEVENTS + .1)
ylim(0.1, 10)
xlabel('Number of Events')
ylabel('Neutron Star Compactness $/\Lambda_\mathrm{true}$')
savefig(os.path.join(plotdir, 'RelErrorBarsLambda_vs_NShifted_AllLambda_2.pdf'))


# In[19]:

# [Same as above, different color scheme, linear y-scale]
# Shown is the quantity (measured - true)/true lambda in line-circled
# curves. The measured value here is the median of the probability distribution
# obtained by accumulating information from N events. About the medians
# are shown actual 90% confidence intervals on the "measured" value,
# normalized to be on the same y-axis as the median. For e.g., take
# the case when the neutron star has lambda = 2000. After 10 events,
# the yellow errorbars range in [-0.25, 0.7], and the median (filled
# yellow circle) is close to 0.125. Therefore, we measured  value is
# 2000 * (1 + 0.125) + (0.7 * 2000) - (0.25 * 2000) = 2250^{+700}_{-500}.
# Grey dotted curves are contours of 1/sqrt(N). Horizontal dashed green lines
# demarkate +-25% and +-50% errors.
linecolor = ['k', 'b', 'g', 'r', 'y', 'k', 'orange']
Narray = np.arange(1, NEVENTS+1)

fig = figure()
ax = fig.add_subplot(111)
for idx, NSL in enumerate(Lambdavec):
    print "---- For Lambda = %f" % NSL
    xx2 = xx[NSL]
    median_array = [xx2.statistical_data[i][0] for i in range(NEVENTS)]
    errorbar_array = [[xx2.statistical_data[i][0] - xx2.statistical_data[i][2],                   xx2.statistical_data[i][3] - xx2.statistical_data[i][0]]  for i in range(NEVENTS)]
    #
    median_array = np.array(median_array) / NSL
    errorbar_array = np.array(errorbar_array) / NSL
    #
    errorbar(Narray, median_array, yerr=np.transpose(errorbar_array),                 fmt=linecolor[idx]+'-o', lw=1+1.2*idx, label='$\Lambda=%.1f$' % NSL, alpha=0.8-0.02*idx)
    hold(True)
    Narray = Narray + 0.1
#
axhline(y=xx2.NSLambda/NSL, color='k', lw=1, alpha=0.75, linestyle='dashed')
axhline(y=xx2.NSLambda/NSL-0.5, color='c', lw=1, alpha=1, linestyle='--')
axhline(y=xx2.NSLambda/NSL+0.5, color='c', lw=1, alpha=1, linestyle='--')
axhline(y=xx2.NSLambda/NSL-0.25, color='c', lw=1, alpha=1, linestyle='dashed')
axhline(y=xx2.NSLambda/NSL+0.25, color='c', lw=1, alpha=1, linestyle='dashed')

ylow, yhigh = ax.get_ylim()
for y_val in np.linspace(ylow + 0.7, yhigh+0.7, num=15):
    ax.plot(np.arange(1, NEVENTS+1, 1), 1 + y_val / np.sqrt(np.arange(1, NEVENTS+1, 1)),        'k', linestyle='dotted', lw=1, alpha=0.4)
    #ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
    #    'k', linestyle='dotted', alpha=0.9, lw=1.2)

#
#ax.set_yscale('log')
grid(False, which='both')
ax.yaxis.grid(False)
ax.xaxis.grid(True)
legend(loc='upper right')
xlim(0.9, NEVENTS + .5)
ylim(0, 3)
xlabel('Number of Events')
ylabel('Neutron Star Compactness $/\Lambda_\mathrm{true}$')
savefig(os.path.join(plotdir, 'RelErrorBarsLambda_vs_NShifted_AllLambda_3.pdf'))


# In[20]:

# Shown is the measured lambda values, the median as line-circled curves, and
# 90% confidence intervals by the filled region's extent, as functions of the 
# number of observed events N. Each color is for a
# different true value for lambda. Corresponding to each filled region is a 
# horizontal dashed line (of the same color) that marks the true lambda value
# which the region should tend towards.

linecolor = ['y', 'b', 'r', 'g', 'k', 'y', 'k', 'orange']
Narray = np.arange(1, NEVENTS+1)

fig = figure()
ax = fig.add_subplot(111)
for idx, NSL in enumerate(Lambdavec):
    print "---- For Lambda = %f" % NSL
    xx2 = xx[NSL]
    median_array = [xx2.statistical_data[i][0] for i in range(NEVENTS)]
    llimit_array = [xx2.statistical_data[i][2] for i in range(NEVENTS)]
    ulimit_array = [xx2.statistical_data[i][3] for i in range(NEVENTS)]
    #
    median_array = np.array(median_array) #/ NSL
    llimit_array = np.array(llimit_array) #/ NSL
    ulimit_array = np.array(ulimit_array) #/ NSL
    #
    fill_between(Narray, llimit_array, y2=ulimit_array,                alpha=0.5-0.05*idx, color=linecolor[idx], label='$\Lambda=%.1f$' % NSL)
    hold(True)
    plot(Narray, median_array, linecolor[idx]+'-o', alpha=0.5)
    
    axhline(y=xx2.NSLambda, color=linecolor[idx], lw=2, alpha=0.5, linestyle='--')
    Narray = Narray + 0.1
#
#axhline(y=xx2.NSLambda, color='k', lw=1, alpha=0.5, linestyle='-')
#axhline(y=xx2.NSLambda - NSL*0.5, color='k', lw=2, alpha=0.5, linestyle='-.')
#axhline(y=xx2.NSLambda + NSL*0.5, color='k', lw=2, alpha=0.5, linestyle='-.')
#axhline(y=xx2.NSLambda - NSL*0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')
#axhline(y=xx2.NSLambda + NSL*0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')
#
#ax.set_yscale('log'
grid(True, which='both')
legend(loc='best')
xlim(0.9, NEVENTS + .5)
#ylim(0, 3)
xlabel('Number of Events')
ylabel('Neutron Star Compactness')
savefig(os.path.join(plotdir, 'FillBetweenRelErrorBarsLambda_vs_NShifted_AllLambda.pdf'))


# In[21]:

# Shown is the measured lambda values normalized by the true value, as 
# functions of the number of observed events $N$. The normalized medians are
# the line-circled curves, and the corresponding $90\%$ confidence intervals 
# are given by the filled region's vertical extent. Each color is for a
# different true value for lambda. Since all measurements are expected to
# approach the true value as more measurements are taken, we expect each
# filled region showing $\Lambda/\Lambda_\mathrm{true}$ should tend towards 1. 
# We expect the rate of the shrinkage of confidence intervals should follow
# 1/sqrt{N}, and therefore show dotted contours of it in grey as well.

# For $\Lambda= \[1500, 200\}$, we see that the information accumulation rate
# approaches $1/\sqrt{N}$ with about $10$ or so observations. Measuring lambda
# from systems of less deformable neutron stars remains more challenging
# and we could plausibly need $10+$ of their observations to bound the 
# measured lambda within $\pm50\%$ of its true value. The median values, on 
# the other hand, approach the true value much faster. It comes within +-25%
# of the true value with only $10-15$ observations. Fowever, for the most compact
# neutron stars we will need $20+$ observations for placing meaningful bounds.

linecolor = ['y', 'r', 'b', 'g', 'c', 'y', 'k', 'orange']
Narray = np.arange(1, NEVENTS+1)

fig = figure()
ax = fig.add_subplot(111)
for idx, NSL in enumerate(Lambdavec):
    print "---- For Lambda = %f" % NSL
    xx2 = xx[NSL]
    median_array = [xx2.statistical_data[i][0] for i in range(NEVENTS)]
    llimit_array = [xx2.statistical_data[i][2] for i in range(NEVENTS)]
    ulimit_array = [xx2.statistical_data[i][3] for i in range(NEVENTS)]
    #
    median_array = np.array(median_array) / NSL
    llimit_array = np.array(llimit_array) / NSL
    ulimit_array = np.array(ulimit_array) / NSL
    #
    fill_between(Narray, llimit_array, y2=ulimit_array,                alpha=0.45-0.05*idx, color=linecolor[idx], label='$\Lambda=%.1f$' % NSL)
    hold(True)
    plot(Narray, median_array, linecolor[idx]+'-o', alpha=0.5)
    #
    Narray = Narray + 0.1
#
axhline(y=xx2.NSLambda/NSL, color='k', lw=1, alpha=0.5, linestyle='-')
axhline(y=xx2.NSLambda/NSL-0.5, color='k', lw=2, alpha=0.5, linestyle='-.')
axhline(y=xx2.NSLambda/NSL+0.5, color='k', lw=2, alpha=0.5, linestyle='-.')
axhline(y=xx2.NSLambda/NSL-0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')
axhline(y=xx2.NSLambda/NSL+0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')
#
ylow, yhigh = ax.get_ylim()
for y_val in np.linspace(ylow + 0.7, yhigh+0.7, num=15):
    ax.plot(np.arange(1, NEVENTS+1, 1), 1 + y_val / np.sqrt(np.arange(1, NEVENTS+1, 1)),        'k', linestyle='dotted', lw=1, alpha=0.5)
    #ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
    #    'k', linestyle='dotted', alpha=0.9, lw=1.2)

#ax.set_yscale('log'
grid(True, which='both')
legend(loc='best')
ax.yaxis.grid(False)
ax.xaxis.grid(True)

xlim(0.9, NEVENTS + .5)
ylim(0, 4)
xlabel('Number of Events')
ylabel('Neutron Star Compactness $/\Lambda_\mathrm{true}$')
savefig(os.path.join(plotdir, 'FillBetweenRelErrorBarsLambda_vs_NShifted_AllLambda.pdf'))


# In[22]:

# [Same as above, but with y-axis on a log scale]
# Shown is the measured lambda values normalized by the true value, as 
# functions of the number of observed events $N$. The normalized medians are
# the line-circled curves, and the corresponding $90\%$ confidence intervals 
# are given by the filled region's vertical extent. Each color is for a
# different true value for lambda. Since all measurements are expected to
# approach the true value as more measurements are taken, we expect each
# filled region showing $\Lambda/\Lambda_\mathrm{true}$ should tend towards 1. 
# We expect the rate of the shrinkage of confidence intervals should follow
# 1/sqrt{N}, and therefore show dotted contours of it in grey as well.

# For $\Lambda= \[1500, 200\}$, we see that the information accumulation rate
# approaches $1/\sqrt{N}$ with about $10$ or so observations. Measuring lambda
# from systems of less deformable neutron stars remains more challenging
# and we could plausibly need $10+$ of their observations to bound the 
# measured lambda within $\pm50\%$ of its true value. The median values, on 
# the other hand, approach the true value much faster. It comes within +-25%
# of the true value with only $10-15$ observations. Fowever, for the most compact
# neutron stars we will need $20+$ observations for placing meaningful bounds.

linecolor = ['b', 'k', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'k', 'b', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'r', 'b', 'g', 'c', 'y', 'k', 'orange']

Narray = np.arange(1, NEVENTS+1)

fig = figure()
ax = fig.add_subplot(111)
for idx, NSL in enumerate(Lambdavec):
    print "---- For Lambda = %f" % NSL
    xx2 = xx[NSL]
    median_array = [xx2.statistical_data[i][0] for i in range(NEVENTS)]
    llimit_array = [xx2.statistical_data[i][2] for i in range(NEVENTS)]
    ulimit_array = [xx2.statistical_data[i][3] for i in range(NEVENTS)]
    #
    median_array = np.array(median_array) / NSL
    llimit_array = np.array(llimit_array) / NSL
    ulimit_array = np.array(ulimit_array) / NSL
    #
    fill_between(Narray, llimit_array, y2=ulimit_array,                alpha=0.5-0.05*idx, color=linecolor[idx], label='$\Lambda=%.1f$' % NSL)
    hold(True)
    plot(Narray, median_array, linecolor[idx]+'-o', alpha=0.5)
    #
    Narray = Narray + 0.1
#
axhline(y=xx2.NSLambda/NSL, color='k', lw=1, alpha=0.5, linestyle='dashed')
axhline(y=xx2.NSLambda/NSL-0.5, color='k', lw=1.5, alpha=0.75, linestyle='-.')
axhline(y=xx2.NSLambda/NSL+0.5, color='k', lw=1.5, alpha=0.75, linestyle='-.')
axhline(y=xx2.NSLambda/NSL-0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')
axhline(y=xx2.NSLambda/NSL+0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')

#
ylow, yhigh = ax.get_ylim()
for y_val in np.linspace(ylow + 0.7, yhigh+0.7, num=10):
    ax.plot(np.arange(1, NEVENTS+1, 1), 1 + y_val / np.sqrt(np.arange(1, NEVENTS+1, 1)),        'k', linestyle='dotted', lw=1, alpha=0.5)
    #ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
    #    'k', linestyle='dotted', alpha=0.9, lw=1.2)

ax.set_yscale('log')
#grid(True, which='both')
legend(loc='best')
ax.yaxis.grid(False)
ax.xaxis.grid(True)
xlim(0.9, NEVENTS + .5)
#ylim(0, 4)
xlabel('Number of Events')
ylabel('Neutron Star Compactness $/\Lambda_\mathrm{true}$')
savefig(os.path.join(plotdir, 'FillBetweenRelErrorBarsLambda_vs_NShifted_AllLambda_Log.pdf'))


# # LAMBDA ERRORS vs LAMBDA
# ## At the start

# In[23]:

######################################################
# Function to make parameter bias plots
######################################################
linestyles = ['-', '--', '-.', '-x', '--o']
linecolors = ['r', 'g', 'b', 'k', 'm', 'y']
gmean = (5**0.5 + 1)/2.

# Figure settings
ppi=72.0
aspect=(5.**0.5 - 1) * 0.5
size=4.0 * 2# was 6
figsize=(size,aspect*size)
plt.rcParams.update({    'legend.fontsize':16,     'text.fontsize':16,    'axes.labelsize':16,    'font.family':'serif',    'font.size':16,    'xtick.labelsize':16,    'ytick.labelsize':16,    'figure.subplot.bottom':0.2,    'figure.figsize':figsize,     'savefig.dpi': 500.0,     'figure.autolayout': True})


# In[24]:

dd = xx[NSL].statistical_data


# In[ ]:




# In[25]:

linecolor = ['b', 'k', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'k', 'b', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'r', 'b', 'g', 'c', 'y', 'k', 'orange']

Narray = np.arange(1, NEVENTS+1)

#fig = figure(figsize=(16,8))
fig = figure()
ax = fig.add_subplot(111)
for n in range(NEVENTS):
    median_array = np.array([xx[NSL].statistical_data[n][0]/NSL for NSL in Lambdavec])
    llimit_array = np.array([xx[NSL].statistical_data[n][2]/NSL for NSL in Lambdavec])
    ulimit_array = np.array([xx[NSL].statistical_data[n][3]/NSL for NSL in Lambdavec])
    #
    #semilogy(Lambdavec, ulimit_array - llimit_array, label='$N=%d$' % n)
    if n < 5:
        lc = 'r'
        alpha = 0.4 - 0.03*n
        lab = '$N=%d$' % (n+1)
    else:
        lc = 'k'
        alpha = 0.2+((n*1.0)/len(range(50)))**2
        if n < 10: lab = '$N=%d$' % (n+1)
        else: lab = ''
    ax.semilogy(Lambdavec, ulimit_array - llimit_array, lc, lw=3, alpha=alpha, label=lab)
    hold(True)
    #plot(Lambdavec, np.zeros(len(Lambdavec)), linecolor[idx]+'-o', alpha=0.5)
    #
ax.axhline(y=1, color='g', lw=1, alpha=0.5, linestyle='dashed')
#axhline(y=xx2.NSLambda/NSL-0.5ulimit_arrayy', lw=1.5, alpha=0.75, linestyle='-.')
#axhline(y=xx2.NSLambda/NSL+0.5, color='k', lw=1.5, alpha=0.75, linestyle='-.')
#axhline(y=xx2.NSLambda/NSL-0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')
#axhline(y=xx2.NSLambda/NSL+0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')
#
ylow, yhigh = ax.get_ylim()
#for y_val in np.linspace(ylow + 0.7, yhigh+0.7, num=10):
#    ax.loglog(np.arange(1, NEVENTS+1, 1), 1 + y_val / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
#        'k', linestyle='dotted', lw=1, alpha=0.5)
    #ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
    #    'k', linestyle='dotted', alpha=0.9, lw=1.2)

ylow, yhigh = ax.get_ylim()
#for y_val in np.linspace(ylow + np.power(500, 1/1.95)-0.8, yhigh+np.power(2000, 1/1.5), num=10):
#    ax.loglog(Lambdavec, 0 + y_val / np.sqrt(Lambdavec),\
#        'g', linestyle='dotted', lw=1, alpha=0.5)
for y_val in np.logspace(np.log10(ylow + np.power(500, 1/1.6)), np.log10(yhigh+np.power(2000, 1/1.03)), num=20):
    ax.hold(True)
    ax.loglog(Lambdavec, 0 + y_val / np.power(Lambdavec, 1/1.2),        'g', linestyle='-.', lw=1, alpha=0.7)#,\
             #label='$y=x^{1/1.2}$')
    #ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
    #    'k', linestyle='dotted', alpha=0.9, lw=1.2)

#ax.text(550, 0.15, '$y=1/x^{1/1.2}$', color='g')
ax.text(550, 0.15, '$y=1/x^{1/(1 + 1/5)}$', color='g')
ax.set_yscale('log')
ax.grid(True, which='major')
ax.legend(loc='best', ncol=2, frameon=False, fontsize=12)
ax.yaxis.grid(False)
ax.xaxis.grid(False)
xlim(500, 2000)
ylim(0.1, 15)
ylabel('Measurement Uncertainty for\n Neutron Star Compactness$/\Lambda_\mathrm{true}$')
xlabel('True Neutron Star Compactness $(\Lambda_\mathrm{true})$')
savefig(os.path.join(plotdir, 'LambdalErrorBars_vs_Lambda_N%d_Log.pdf' % n))


# ## Get the slope of best fit line for lambda uncertainties

# In[26]:

slopes = np.array([])
for n in range(NEVENTS):
    median_array = np.array([xx[NSL].statistical_data[n][0]/1 for NSL in Lambdavec])
    llimit_array = np.array([xx[NSL].statistical_data[n][2]/1 for NSL in Lambdavec])
    ulimit_array = np.array([xx[NSL].statistical_data[n][3]/1 for NSL in Lambdavec])
    xvec= Lambdavec
    yvec= ulimit_array - llimit_array
    slope, offset = np.polyfit(np.log10(xvec), np.log10(yvec), 1)
    slopes = np.append(slopes, slope)


# In[27]:

figure()
plot(range(NEVENTS), slopes, 'k', lw=4, alpha=0.6)
grid(True, which='both')
ylabel('Coefficient $\\beta:\,\\delta\\Lambda_\\mathrm{NS}\propto\\Lambda_\\mathrm{NS}^\\beta$')
xlabel('Number of Events')
axhline(1/5., color='g', linestyle='dashed', lw=3, alpha=0.5)
text(30, 0.1, '$\\delta\\Lambda_\\mathrm{NS}\propto\\Lambda_\\mathrm{NS}^{1/5}$', color='g', alpha=0.5)
savefig(os.path.join(plotdir, 'PowerLawCoefficient_LambdaErrorvsLambda_vs_N.pdf'))


# In[28]:

Nvec = np.arange(NEVENTS)

for NSL in Lambdavec:    
    slopes = np.array([])
    for n in Nvec:
        #
        nvec = np.arange(1+n)
        ## Get lambda errors and medians for all N events cumulatively, for fixed Lambda
        median_array = np.array([xx[NSL].statistical_data[i][0]/1 for i in nvec])
        llimit_array = np.array([xx[NSL].statistical_data[i][2]/1 for i in nvec])
        ulimit_array = np.array([xx[NSL].statistical_data[i][3]/1 for i in nvec])
        #
        ## Get X array, i.e. N, and Y array, i.e. Lambda error vs N
        xvec= 1. + nvec
        yvec= ulimit_array - llimit_array
        #
        ## Now find the dependence of Lambda errors on N
        slope, offset = np.polyfit(np.log10(xvec), -1*np.log10(yvec), 1)
        slopes = np.append(slopes, slope)
    #
    #if n < 10: lab = '$N=%d$' % (n+1)
    #if n < 5: lc, alpha = 'r', 0.3
    #else: lc, alpha = 'k', 0.1 + 0.9 * n/NEVENTS
    lc = 'k'
    alpha = 0.25 + 0.6 * (NSL-500)/2000.0
    lab = '$\\Lambda_\\mathrm{NS}=%d$' % NSL
    #
    ## Plot the exponent for each Lambda
    plot(Nvec, slopes, lc, lw=4, alpha=alpha, label=lab)
    hold(True)

grid(True, which='both')
ylabel('Coefficient $\\alpha:\,\\delta\\Lambda_\\mathrm{NS}\propto N^{-\\alpha}$')
xlabel('Number of Events')
legend(loc='best')
axhline(1/2., color='g', linestyle='dashed', lw=3, alpha=0.5)
text(50, 0.3, '$\\delta\\Lambda_\\mathrm{NS}\propto 1/\sqrt{N}$', color='g', alpha=0.5)

#xlim(0,1)
savefig(os.path.join(plotdir, 'PowerLawCoefficient_LambdaErrorvsN_vs_Lambda.pdf'))


# ## Now show the Median

# In[32]:

linecolor = ['b', 'k', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'k', 'b', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'r', 'b', 'g', 'c', 'y', 'k', 'orange']

Narray = np.arange(1, NEVENTS+1)

fig = figure()
ax = fig.add_subplot(111)
for n in range(NEVENTS):
    median_array = np.array([xx[NSL].statistical_data[n][0]/NSL for NSL in Lambdavec])
    llimit_array = np.array([xx[NSL].statistical_data[n][2]/NSL for NSL in Lambdavec])
    ulimit_array = np.array([xx[NSL].statistical_data[n][3]/NSL for NSL in Lambdavec])
    #
    #semilogy(Lambdavec, ulimit_array - llimit_array, label='$N=%d$' % n)
    if n < 5:
        lc = 'r'
        alpha = 0.1+((n*1.0)/len(range(5)))**2
        lab = '$N=%d$' % (n+1)
    else:
        lc = 'k'
        alpha = 0.2+((n*1.0)/len(range(50)))**2
        if n < 10: lab = '$N=%d$' % (n+1)
        else: lab = ''
    semilogy(Lambdavec, np.abs(median_array-1), lc, lw=2.5, alpha=alpha, label=lab)
    hold(True)
    #plot(Lambdavec, np.zeros(len(Lambdavec)), linecolor[idx]+'-o', alpha=0.5)
    #
#axhline(y=0.1, color='g', lw=1, alpha=0.5, linestyle='dashed')
axhline(y=1./10., color='brown', lw=4, alpha=0.5, linestyle='dashed')
ax.text(1750, 0.12, "$10\%\, \mathrm{error}$", color='brown', alpha=0.5)

#axhline(y=xx2.NSLambda/NSL-0.5ulimit_arrayy', lw=1.5, alpha=0.75, linestyle='-.')
#axhline(y=xx2.NSLambda/NSL+0.5, color='k', lw=1.5, alpha=0.75, linestyle='-.')
#axhline(y=xx2.NSLambda/NSL-0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')
#axhline(y=xx2.NSLambda/NSL+0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')
#
ylow, yhigh = ax.get_ylim()
#for y_val in np.linspace(ylow + 0.7, yhigh+0.7, num=10):
#    ax.loglog(np.arange(1, NEVENTS+1, 1), 1 + y_val / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
#        'k', linestyle='dotted', lw=1, alpha=0.5)
    #ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
    #    'k', linestyle='dotted', alpha=0.9, lw=1.2)

ax.set_yscale('log')
grid(True, which='both')
legend(loc='lower right', ncol=3, frameon=True, fontsize=12)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
xlim(500, 2000)
ylim(0.001, 5)
ylabel('$|$Median Neutron Star\n Compactness $\,/\Lambda_\mathrm{true} -1|$')
#ylabel('Measurement Uncertainty for\n Neutron Star Compactness$/\Lambda_\mathrm{true}$')
xlabel('True Neutron Star Compactness $(\Lambda_\mathrm{true})$')
savefig(os.path.join(plotdir, 'RelErrLambdaMedian_vs_Lambda_N_Log.pdf'))


# # LAMBDA ERRORS vs NO OF EVENTS

# In[35]:

# In the main panel is shown the measurement uncertainty for lambda
# at $90\%$ confidence level, normalized by the true value of lambda,
# as a function of the number of events observed. In the inset, we
# show the same measurement uncertainty, but not normalized, as a function
# of $N$.

# First by looking at the region with $<5$ events in both the inset and
# the main plot, we find that for the first $3-4$ events, our measurement is
# prior dominated. That is, the $90\%$ confidence interval spans the entire
# range of allowed values in $[0, 4000]$. 

linecolor = ['b', 'k', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'k', 'b', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'r', 'b', 'g', 'c', 'y', 'k', 'orange']

Narray = np.arange(1, NEVENTS+1)

fig = figure()
ax = fig.add_subplot(111)
for NSL in Lambdavec:
    median_array = np.array([xx[NSL].statistical_data[n][0]/NSL for n in range(NEVENTS)])
    llimit_array = np.array([xx[NSL].statistical_data[n][2]/NSL for n in range(NEVENTS)])
    ulimit_array = np.array([xx[NSL].statistical_data[n][3]/NSL for n in range(NEVENTS)])
    #
    lc = 'g'
    alpha = NSL*1.0 / max(Lambdavec)
    lab = '$\Lambda_\mathrm{true}=%d$' % (NSL)
    loglog(Narray, ulimit_array - llimit_array, lc, lw=3. -0* 1.5*NSL/max(Lambdavec), alpha=alpha, label=lab)
    hold(True)
    #plot(Lambdavec, np.zeros(len(Lambdavec)), linecolor[idx]+'-o', alpha=0.5)
    #
#
ylow, yhigh = ax.get_ylim()
for y_val in np.linspace(ylow + 0.7, yhigh+0.7, num=10):
    ax.loglog(Narray, 0 + y_val / np.sqrt(Narray),        'k', linestyle='dotted', lw=1, alpha=0.5)
    #ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
    #    'k', linestyle='dotted', alpha=0.9, lw=1.2)

ax.text(15, 0.15, "$y=1/x^{1+1/5}$", alpha=0.65)
ax.set_yscale('log')
#grid( which='both')
legend(loc='upper right')
ax.yaxis.grid(False)
ax.xaxis.grid(True, which='both')
xlim(min(Narray)-0.1, max(Narray)+0.1)
#ylim(100, 10000)
ylim(1/10., 10.)
ylabel('Measurement Uncertainty for\n Neutron Star Compactness$/\Lambda_\mathrm{true}$')
xlabel('Number of Events')

## Add the recovered lambda plot
a = plt.axes([.21, .21, .35, .3], axisbg='w')
for NSL in Lambdavec:
    median_array = np.array([xx[NSL].statistical_data[n][0]/1 for n in range(NEVENTS)])
    llimit_array = np.array([xx[NSL].statistical_data[n][2]/1 for n in range(NEVENTS)])
    ulimit_array = np.array([xx[NSL].statistical_data[n][3]/1 for n in range(NEVENTS)])
    #
    lc = 'm'
    alpha = NSL*1.0 / max(Lambdavec)
    lab = '$\Lambda_\mathrm{true}=%d$' % (NSL)
    plt.loglog(Narray, ulimit_array - llimit_array, lc, lw=3. -0* 1.5*NSL/max(Lambdavec), alpha=alpha, label=lab)
    hold(True)
    #plot(Lambdavec, np.zeros(len(Lambdavec)), linecolor[idx]+'-o', alpha=0.5)
    #
ylow, yhigh = a.get_ylim()
for y_val in np.linspace(ylow + 0.7, yhigh+0.7, num=10):
    a.loglog(Narray, 0 + y_val / np.sqrt(Narray),        'k', linestyle='dotted', lw=1, alpha=0.5)

plt.ylim(100,10000)
plt.xlim(min(Narray)-0.1, max(Narray)+0.1)
a.yaxis.grid(False)
a.xaxis.grid(True, which='both')

hold(True)
savefig(os.path.join(plotdir, 'LambdaCIWidths_vs_N_Log.pdf'))


# In[36]:

# In the main panel is shown the measurement uncertainty for lambda
# at $90\%$ confidence level, normalized by the true value of lambda,
# as a function of the number of events observed. In the inset, we
# show the same measurement uncertainty, but not normalized, as a function
# of $N$.

# First by looking at the region with $<5$ events in both the inset and
# the main plot, we find that for the first $3-4$ events, our measurement is
# prior dominated. That is, the $90\%$ confidence interval spans the entire
# range of allowed values in $[0, 4000]$. 

linecolor = ['b', 'k', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'k', 'b', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'r', 'b', 'g', 'c', 'y', 'k', 'orange']

Narray = np.arange(1, NEVENTS+1)

fig = figure()
ax = fig.add_subplot(111)
for NSL in Lambdavec:
    median_array = np.array([xx[NSL].statistical_data[n][0]/NSL for n in range(NEVENTS)])
    llimit_array = np.array([xx[NSL].statistical_data[n][2]/NSL for n in range(NEVENTS)])
    ulimit_array = np.array([xx[NSL].statistical_data[n][3]/NSL for n in range(NEVENTS)])
    #
    lc = 'g'
    alpha = NSL*1.0 / max(Lambdavec)
    lab = '$\Lambda_\mathrm{true}=%d$' % (NSL)
    semilogy(Narray, ulimit_array - llimit_array, lc, lw=3. -0* 1.5*NSL/max(Lambdavec), alpha=alpha, label=lab)
    hold(True)
    #plot(Lambdavec, np.zeros(len(Lambdavec)), linecolor[idx]+'-o', alpha=0.5)
    #
#
ax.axhline(y=2, linestyle='dashed', color='brown', lw=2, alpha=0.8)
ax.axhline(y=1, linestyle='dashed', color='brown', lw=2, alpha=0.8)
ax.axhline(y=0.5, linestyle='dashed', color='brown', lw=2, alpha=0.8)

ylow, yhigh = ax.get_ylim()
for y_val in np.linspace(ylow + 0.7, yhigh-0.7, num=15):
    ax.semilogy(Narray, 0 + y_val / np.sqrt(Narray),        'k', linestyle='dotted', lw=1, alpha=0.5)
    #ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
    #    'k', linestyle='dotted', alpha=0.9, lw=1.2)

ax.text(12, 3, "$y=1/x^{1+1/5}$", alpha=0.65)
ax.set_yscale('log')
#grid( which='both')
legend(loc='lower left', ncol=3, frameon=False, fontsize=12)
ax.yaxis.grid(False)
ax.xaxis.grid(True, which='both')
xlim(min(Narray)-0.1, max(Narray)+0.1)
#ylim(100, 10000)
ylim(0.09, 10.)
ylabel('Measurement Uncertainty for\n Neutron Star Compactness$/\Lambda_\mathrm{true}$')
xlabel('Number of Events')

## Add the recovered lambda plot
a = plt.axes([.57, .6, .35, .3], axisbg='w')
for NSL in Lambdavec:
    median_array = np.array([xx[NSL].statistical_data[n][0]/1 for n in range(NEVENTS)])
    llimit_array = np.array([xx[NSL].statistical_data[n][2]/1 for n in range(NEVENTS)])
    ulimit_array = np.array([xx[NSL].statistical_data[n][3]/1 for n in range(NEVENTS)])
    #
    lc = 'm'
    alpha = NSL*1.0 / max(Lambdavec)
    lab = '$\Lambda_\mathrm{true}=%d$' % (NSL)
    plt.loglog(Narray, ulimit_array - llimit_array, lc,                 lw=3. -0* 1.5*NSL/max(Lambdavec), alpha=alpha, label=lab)
    hold(True)
    #plot(Lambdavec, np.zeros(len(Lambdavec)), linecolor[idx]+'-o', alpha=0.5)
    #
ylow, yhigh = a.get_ylim()
for y_val in np.linspace(ylow - 00, yhigh+0.7, num=6):
    a.loglog(Narray, 0 + y_val / np.sqrt(Narray),        'k', linestyle='dotted', lw=1, alpha=0.5)

plt.ylim(90,10000)
plt.xlim(min(Narray)-0.1, max(Narray)+0.1)
a.yaxis.grid(False)
a.xaxis.grid(True, which='both')

hold(True)
savefig(os.path.join(plotdir, 'LambdaCIWidths_vs_N.pdf'))


# ## Show the MEDIAN

# In[37]:

linecolor = ['b', 'k', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'k', 'b', 'g', 'r', 'y', 'k', 'orange']
linecolor = ['y', 'r', 'b', 'g', 'c', 'y', 'k', 'orange']

Narray = np.arange(1, NEVENTS+1)

fig = figure()
ax = fig.add_subplot(111)
for NSL in Lambdavec:
    median_array = np.array([xx[NSL].statistical_data[n][0]/NSL for n in range(NEVENTS)])
    llimit_array = np.array([xx[NSL].statistical_data[n][2]/NSL for n in range(NEVENTS)])
    ulimit_array = np.array([xx[NSL].statistical_data[n][3]/NSL for n in range(NEVENTS)])
    #
    lc = 'blue'
    alpha = NSL*1.0 / max(Lambdavec)
    lab = '$\Lambda_\mathrm{true}=%d$' % (NSL)
    plot(Narray, np.abs(median_array-1), lc, lw=3. -0* 1.5*NSL/max(Lambdavec), alpha=alpha, label=lab)
    hold(True)
    #plot(Lambdavec, np.zeros(len(Lambdavec)), linecolor[idx]+'-o', alpha=0.5)
    #
axhline(y=1./10., color='brown', lw=4, alpha=0.5, linestyle='dashed')
ax.text(40, 0.12, "$10\%\, \mathrm{error}$", color='brown', alpha=0.5)
#axhline(y=xx2.NSLambda/NSL-0.5ulimit_arrayy', lw=1.5, alpha=0.75, linestyle='-.')
#axhline(y=xx2.NSLambda/NSL+0.5, color='k', lw=1.5, alpha=0.75, linestyle='-.')
#axhline(y=xx2.NSLambda/NSL-0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')
#axhline(y=xx2.NSLambda/NSL+0.25, color='k', lw=1.5, alpha=0.75, linestyle='dashed')
#
ylow, yhigh = ax.get_ylim()
#for y_val in np.linspace(ylow + 0.7, yhigh+0.7, num=10):
#    ax.loglog(np.arange(1, NEVENTS+1, 1), 1 + y_val / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
#        'k', linestyle='dotted', lw=1, alpha=0.5)
    #ax2.plot(np.arange(1, NEVENTS+1, 1), NSL + y_val * scale_fac / np.sqrt(np.arange(1, NEVENTS+1, 1)),\
    #    'k', linestyle='dotted', alpha=0.9, lw=1.2)

ax.set_yscale('log')
grid(True, which='both')
legend(loc='best', ncol=3, frameon=False)
ax.yaxis.grid(True)
ax.xaxis.grid(True)
xlim(min(Narray)-0.1, max(Narray)+0.1)
ylim(0, 1.7)
ylabel('$|$Median Neutron Star\n Compactness $\,/\Lambda_\mathrm{true} -1|$')
xlabel('Number of Events')
savefig(os.path.join(plotdir, 'RelErrorLambdaMedian_vs_N.pdf'))


# In[ ]:



