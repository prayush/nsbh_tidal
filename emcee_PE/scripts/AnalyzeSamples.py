#!/usr/bin/env python
# Tools for analysis of MCMC samples from emcee for spin estimation project
# Copyright 2015
# 
__author__ = "Prayush Kumar <prayush.kumar@ligo.org>"

import os, sys
import commands as cmd
import numpy as np
from matplotlib import mlab, cm, use
#use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex' : True})
from mpl_toolkits.axes_grid1 import ImageGrid

from pydoc import help
from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde
from scipy import stats
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import minimize_scalar

sys.path.append( os.path.dirname(os.path.realpath(__file__)) + '/../src/' )
sys.path.append( os.path.dirname(os.path.realpath(__file__)) )
sys.path.append( '/home/prayush/src/UseNRinDA/scripts/' )
sys.path.append( '/home/prayush/src/UseNRinDA/plotting' )

from utils import *
import pycbc.pnutils as pnutils
from PlotOverlapsNR import make_contour_plot_multrow
import h5py


# Constants
verbose  = True
vverbose = False
debug    = False

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
linestyles = ['-', '-.', '--', ':', '--o']
linecolors = ['r', 'olivedrab', 'k', 'b', 'm', 'y']
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
'savefig.dpi': 300.0, \
'figure.autolayout': True})

# hist(PercentileInterval(s1[0], pc=99.0), 100);


# Plotting functions
def make_contour_array(X, Y, Z2d, xlabel='Time (s)', ylabel='', clabel='', \
        title='', titles=[], \
        levelspacing=0.25, vmin=None, vmax=None, cmap=cm.rainbow, cfmt='%.1f',\
        xmin=None, xmax=None, ymin=None, ymax=None,\
        colorbartype='simple', figname='plot.png'):
  """
  Function to plot arbitrary numbers of contour plots in a single figure
  """
  if colorbartype != 'simple':
    raise IOError("Colorbar type %s not supported" % colorbartype)
  colwidth = 1.7
  if np.shape(X)[:2] != np.shape(Y)[:2] or np.shape(X)[:2] != np.shape(Z2d)[:2]:
    raise IOError("X, Y and Z arrays have different number of sets to plot")
  
  plt.rcParams.update({\
    'legend.fontsize':16, \
    'text.fontsize':16,\
    'axes.labelsize':16,\
    'font.family':'serif',\
    'font.size':16,\
    'xtick.labelsize':16,\
    'ytick.labelsize':16,\
    'figure.subplot.bottom':0.2,\
    'figure.figsize':figsize, \
    'savefig.dpi': 300.0, \
    'figure.autolayout': True})

  nrow, ncol, _ = np.shape(X)
  if vverbose: print "Making plot with %d rows, %d cols" % (nrow, ncol)
  pltid = 0
  
  fig = plt.figure(int(1e7 * np.random.random()), \
              figsize=((2.1*gmean*ncol+1.25)*colwidth, 1.2*colwidth*nrow))
  fig.clf()
  grid = ImageGrid(fig, 111, nrows_ncols=(nrow, ncol), \
            share_all=True,\
            cbar_mode="single", cbar_location="right",\
            cbar_pad=0.05, cbar_size="2%", \
            aspect=True,\
            add_all=True)
  # Find the maximum Z-value of all subplots
  #VMIN, VMAX = np.inf, -np.inf
  #for idx in range(nrow):
  #  for jdx in range(ncol):
  #    VMIN = min(VMIN, Z2d[idx][jdx].min())
  #    VMAX = max(VMAX, Z2d[idx][jdx].max())
  VMINmax, VMAXmin = np.array([]), np.array([])
  for idx in range(nrow):
    for jdx in range(ncol):
      VMINmax = np.append( VMINmax, Z2d[idx][jdx].min() )
      VMAXmin = np.append( VMAXmin, Z2d[idx][jdx].max() )
  VMIN, VMAX = VMINmax.min(), VMAXmin.max()
  print "VMIN = %e, VMAX = %e" % (VMIN, VMAX)
  if VMIN > VMAX: raise IOError("Cant fit all data on a single colorbar")
  
  # Now make contour plots
  for idx in range(nrow):
    for jdx in range(ncol):
      try: xx, yy, zz = X[idx][jdx], Y[idx][jdx], Z2d[idx][jdx]
      except: 
        print "Array shapes = ", np.shape(xx), np.shape(yy), np.shape(zz)
        raise RuntimeError
      if vverbose: print "Adding subplot %d of %d" % (pltid+1, nrow*ncol)
      if debug:
        print "Plotting ", xx, " and ", yy, " versus ", zz
      ax = grid[pltid]
      ax.set_aspect(1./4/gmean)
      pltid += 1
      
      norm = cm.colors.Normalize(vmax=VMAX, vmin=VMIN)#zz.min())
      #cmap = cm.rainbow
      #levels = np.arange(zz.min(), zz.max(), levelspacing)
      levels = np.arange(VMIN, VMAX, levelspacing)
      CS = ax.contourf( xx, yy, zz,\
              levels=levels, \
              cmap = cm.get_cmap(cmap, len(levels)-1), norm=norm,\
              alpha=0.9, vmin=VMIN, vmax=VMAX)
      ax.grid()
      ax.set_xlim([xmin, xmax])
      ax.set_ylim([ymin, ymax])
      if idx == (nrow-1): ax.set_xlabel(xlabel)
      if jdx == 0: ax.set_ylabel(ylabel)
      #if np.shape(titles) == (nrow, ncol): ax.set_title(titles[idx][jdx])
      if np.shape(titles) == (nrow, ncol):
        ax.text(.5, .9, titles[idx][jdx], horizontalalignment='center', transform=ax.transAxes)
      #if idx == 0 and jdx==(ncol/2) and title != '':
      #  ax.set_title(titles[idx][jdx]+'\n '+title)  
      if idx == 0 and jdx==(ncol/2) and title != '':
        ax.text(.5, .9, titles[idx][jdx]+'\n '+title, horizontalalignment='center', transform=ax.transAxes)
  #
  cb = ax.cax.colorbar(CS, format=cfmt)
  cb.set_label_text(clabel)
  ax.cax.toggle_label(True)
  fig.subplots_adjust(right=0.8)
  #fig.tight_layout(rect=(0, 0, 0.82, 1))
  fig.savefig(figname)
  return  


# Plotting functions
def make_contour_array_old(X, Y, Z2d, xlabel='Time (s)', ylabel='', clabel='', \
                title='', titles=[], \
                levelspacing=0.25, vmin=None, vmax=None,\
                xmin=None, xmax=None, ymin=None, ymax=None,\
                colorbartype='simple', figname='plot.png'):
  print "DONT USE THIS FUNCTION, ITS DEPRECATED!!!"
  return
  if colorbartype != 'simple':
    raise IOError("Colorbar type %s not supported" % colorbartype)
  colwidth = 2.
  if np.shape(X)[:2] != np.shape(Y)[:2] or np.shape(X)[:2] != np.shape(Z2d)[:2]:
    raise IOError("X, Y and Z arrays have different number of sets to plot")
  
  nrow, ncol, _ = np.shape(X)
  if vverbose: print "Making plot with %d rows, %d cols" % (nrow, ncol)
  pltid = 1
  frac = (2.*gmean*ncol+0.3)
  frac = ncol
  fig = plt.figure(int(1e7 * np.random.random()), \
              figsize=(frac*colwidth, colwidth*nrow))
  fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=True,\
                    figsize=((2.*gmean*ncol+0.3)*colwidth, colwidth*nrow))
  for idx in range(nrow):
    for jdx in range(ncol):
      try: xx, yy, zz = X[idx][jdx], Y[idx][jdx], Z2d[idx][jdx]
      except: 
        print "Array shapes = ", np.shape(xx), np.shape(yy), np.shape(zz)
        raise RuntimeError
      if vverbose: print "Adding subplot %d of %d" % (pltid, nrow*ncol)
      #ax = fig.add_subplot(nrow, ncol, pltid, autoscale_on=True)
      ax = axes[idx][jdx]
      pltid += 1
      
      norm = cm.colors.Normalize(vmax=zz.max(), vmin=zz.min())
      cmap = cm.rainbow
      levels = np.arange(zz.min(), zz.max(), levelspacing)
      CS = ax.contourf( xx, yy, zz,\
              levels=levels, \
              cmap = cm.get_cmap(cmap, len(levels)-1), norm=norm,\
              alpha=0.9, vmin=vmin, vmax=vmax)
      ax.grid()
      ax.set_xlim([xmin, xmax])
      ax.set_ylim([ymin, ymax])
      if idx == (nrow-1): ax.set_xlabel(xlabel)
      if jdx == 0: ax.set_ylabel(ylabel)
      if np.shape(titles) == (nrow, ncol): ax.set_title(titles[idx][jdx])
      if idx == 0 and jdx==(ncol/2) and title != '':
        ax.set_title(titles[idx][jdx]+'\n '+title)      
  #
  fig.subplots_adjust(right=0.8)
  if colorbartype=='simple':
    ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.7])
    cb = fig.colorbar(CS, cax=ax2, orientation=u'vertical', format='%.1f')
    cb.set_label(clabel)
    cb.set_clim(vmin=vmin, vmax=vmax)
    fig.subplots_adjust(right=0.8)
  else:
    # The following code is copied and might not work
    ax2 = fig.add_axes([0.92, 0.1, 0.01, 0.7])
    #ax2 = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    # Make the colorbar
    if type(bounds) == np.ndarray or type(bounds) == list:
      cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
            spacing='uniform', format='%.2f', orientation=u'vertical',\
            ticks=bounds, boundaries=bounds)
    else:
      # How does this colorbar know what colors to span??
      cb = mp.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, \
            spacing='uniform', format='%.2f', orientation=u'vertical',\
            ticks=tmp_bounds)
    # Add tick labels
    if logC and (type(bounds) == np.ndarray or type(bounds) == list):
      cb.set_ticklabels(np.round(10**bounds, decimals=4))
    elif type(bounds) == np.ndarray or type(bounds) == list:
      cb.set_ticklabels(np.round(bounds, decimals=4))
    #cb = fig.colorbar(scat, shrink=0.5, aspect=30, spacing='proportional',\
    #  ticks=[0,-0.5,-1,-1.3,-1.6,-1.9,-2.1,-2.4,-2.7,-3])
    #ax2.set_title(clabel, loc='left')
    cb.set_label(clabel, verticalalignment='top', horizontalalignment='center')#,labelpad=-0.3,y=1.1,x=-0.5)  
    #if max(C) < cmax and min(C) > cmin: cb.set_clim([cmin,cmax])

  #fig.tight_layout(rect=(0,0,0.93,1))

  fig.savefig(figname)
  return



def make_contour(X, Y, Z2d, xlabel='Time (s)', ylabel='', clabel='', title='',\
                levelspacing=0.25, vmin=None, vmax=None, cmap=cm.Reds_r,\
                xmin=None, xmax=None, ymin=None, ymax=None,\
                cbfmt='%.1f', figname='plot.png'):
  colwidth = 4.8
  plt.figure(int(1e7 * np.random.random()), figsize=(1.*gmean*colwidth, colwidth))
  norm = cm.colors.Normalize(vmax=Z2d.max(), vmin=Z2d.min())
  #cmap = cm.rainbow
  levels = np.arange(Z2d.min(), Z2d.max(), levelspacing)
  plt.contourf( X, Y, Z2d,\
              levels=levels, \
              cmap = cm.get_cmap(cmap, len(levels)-1), norm=norm,\
              alpha=0.9, vmin=vmin, vmax=vmax)
  plt.grid()
  plt.xlim([xmin,xmax])
  plt.ylim([ymin,ymax])
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  cb = plt.colorbar(format=cbfmt)
  cb.set_label(clabel)
  cb.set_clim(vmin=vmin, vmax=vmax)
  plt.savefig(figname)
  return


def make_scatter(X, Y, Z, xlabel='', ylabel='', clabel='', vmin=-1, vmax=None,\
                  figname='plot.png'):
  # Just plot the data
  plt.figure(figsize=(18,6))
  #plt.hexbin(X, Y, C=Z, gridsize=200)
  plt.scatter(X, Y, c=Z, linewidths=0, vmin=vmin, vmax=vmax)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  cb = plt.colorbar()
  cb.set_label(clabel)
  cb.set_clim(vmin=vmin, vmax=vmax)
  plt.grid()
  plt.savefig(figname,dpi=400)

def make_hexbin(X, Y, Z, xlabel='', ylabel='', clabel='', vmin=-1, vmax=None,\
                  figname='plot.png'):
  # Just plot the data
  plt.figure(figsize=(18,6))
  #plt.hexbin(X, Y, C=Z, gridsize=200)
  plt.hexbin(X, Y, C=Z, vmin=vmin, vmax=vmax)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  cb = plt.colorbar()
  cb.set_label(clabel)
  cb.set_clim(vmin=vmin, vmax=vmax)
  plt.grid()
  plt.savefig(figname,dpi=400)


def make_multilines(XY, labels=None, xlabel='SNR', ylabel='', clabel='', title='',\
                levelspacing=0.25, vmin=None, vmax=None, cmap=cm.Reds_r,\
                pcolors=linecolors, pmarkers=linestyles,\
                xmin=None, xmax=None, ymin=None, ymax=None,\
                single_legend=True,\
                cbfmt='%.1f', figname='plot.png'):
  #
  colwidth = 4.8  
  plt.rcParams.update({\
    'legend.fontsize':12, \
    'text.fontsize':16,\
    'axes.labelsize':16,\
    'font.family':'serif',\
    'font.size':16,\
    'xtick.labelsize':16,\
    'ytick.labelsize':16,\
    'figure.subplot.bottom':0.2,\
    'figure.figsize':figsize, \
    'savefig.dpi': 300.0, \
    'figure.autolayout': True})
  
  ngrp1 = len(XY.keys())
  ngrp2 = len(XY[XY.keys()[0]].keys())
  if ngrp1 > len(pcolors) or ngrp2 > len(pmarkers):
    raise IOError("More lines to be made than colors/markers given")
  if vverbose: print "Making plot with %d groups of %d lines" % (ngrp1, ngrp2)
    
  #fig = plt.figure(int(1e7 * np.random.random()), \
  #            figsize=((2.1*gmean*ncol+1.25)*colwidth, 1.2*colwidth*nrow))
  plt.figure(int(1e7 * np.random.random()), figsize=(1.*gmean*colwidth, colwidth))
  
  # FIrst make all lines
  all_lines = []
  for i, ki in enumerate(XY.keys()):
    grp_lines = []
    for j, kj in enumerate(XY[ki].keys()):
      X, Y = XY[ki][kj]
      line, = plt.plot(X, Y, c=pcolors[i], ls=pmarkers[j], lw=1.5, markersize=7, label=labels[ki][kj])
      grp_lines.append( line )
    all_lines.append( grp_lines )
  #
  # Now make legends, one to indicate each group's characteristic
  if single_legend:
    harray = [all_lines[i][0] for i in range(ngrp1)]
    harray.extend( [all_lines[0][i] for i in range(ngrp2)] )
    first_legend = plt.legend(handles=harray, loc=1, ncol=2, framealpha=False)
    ax = plt.gca().add_artist(first_legend)
  else:
    harray1 = [all_lines[i][0] for i in range(ngrp1)]
    harray2 = [all_lines[0][i] for i in range(ngrp2)]
    first_legend = plt.legend(handles=harray1, loc=1, ncol=1, framealpha=False)
    ax = plt.gca().add_artist(first_legend)
    second_legend = plt.legend(handles=harray2, loc=2, ncol=1, framealpha=False, markerfirst=False)
  #
  plt.grid(alpha=0.5)
  plt.xlim([xmin,xmax])
  plt.ylim([ymin,ymax])
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.savefig(figname)
  return



def get_simdirname(q, mNS, chi2, Lambda, SNR, Nw, Ns):
  #{{{
  return 'q%.2f_mNS%.2f_chiBH%.2f_Lambda%.1f_SNR%.1f/NW%d_NS%d'\
              % (q, mNS, chi2, Lambda, SNR, Nw, Ns)
  #}}}

def make_bias_plot(plotparam='Mc', plotquantity='bias', normalize='',\
                    xlabel='$\\rho$', ylabel=None,\
                    savefig='plots/plot.png',\
                    xmin=None, xmax=None, ymin=None, ymax=None):
  #{{{
  colwidth = 2.8
  # parameter bias
  plt.figure(int(1e7 * np.random.random()), figsize=(2.*gmean*colwidth, colwidth))
  for pltid, q in enumerate(qvec):
    plt.subplot(1, len(qvec), pltid + 1)
    for x2idx, chi2 in enumerate(chi2vec):
      for Lidx, Lambda in enumerate(Lambdavec):
        if 'Lambda' in normalize: norm = Lambda
        #elif 'eta' in normalize: norm = q / (1. + q)**2.
        #elif 'Mc' in normalize:
        #  tmp_mt = mNS * (1. + q)
        #  norm = tmp_mt * (q / (1. + q)**2.)**0.6
        else: norm = 1.
        param_bias, param_CIwidth90 = {}, {}
        for SNR in SNRvec:
          for Ns in Nsamples:
            for Nw in Nwalkers:
              simdir = get_simdirname(q, mNS, chi2, Lambda, SNR, Nw, Ns)
              print "\n\n Trying to read in %s" % simdir
              chi1val, chi2val = chi1, chi2
              #
              # Hack for NS templates
              if recover_tidal: chi1val, chi2val = chi2, Lambda
              #
              bias, _, width90 = get_bias(basedir=cmd.getoutput('pwd -P'),\
                      simdir=simstring+simdir+'/',\
                      M_inj=(1.+q)*mNS, eta_inj=q/(1.+q)**2,\
                      chi1_inj=chi1val, chi2_inj=chi2val)
              if bias == None:
                print >> sys.stdout, "Data for %s not found. Ignoring.."\
                                        % (simstring+simdir)
                continue
              for kk in bias:
                if kk in param_bias: param_bias[kk].append(bias[kk])
                else: param_bias[kk] = [bias[kk]]
                if kk in param_CIwidth90: param_CIwidth90[kk].append(width90[kk])
                else: param_CIwidth90[kk] = [width90[kk]]
        # Make the plot legend
        if pltid == 0:
          if Lidx == 0:
            # this happens when x2idx's value changes, ie once for each linestyle
            labelstring = '$\chi_{BH} = %.1f$' % (chi2)
          else: labelstring=''
        elif pltid == 1:
          if x2idx == 0:
            # this happens when Lidx = 0,1,2, i.e. once for each linecolor
            labelstring = '$\Lambda = %.0f$' % (Lambda)
          else: labelstring = ''
        # Make the acual plot
        if plotquantity == 'bias': 
          plotvec = np.array(param_bias[plotparam])/norm
        elif plotquantity == 'width90' or 'width' in plotquantity: 
          plotvec = np.array(param_CIwidth90[plotparam])/norm
        if labelstring == '':
          plt.plot(SNRvec, plotvec, linecolors[Lidx]+linestyles[x2idx],lw=2)
        else:
          plt.plot(SNRvec, plotvec,\
                  linecolors[Lidx]+linestyles[x2idx],\
                  label=labelstring,lw=2)      
    #
    plt.legend(loc='best')
    plt.grid()
    if xlabel: plt.xlabel(xlabel)
    if pltid == 0 and ylabel: plt.ylabel(ylabel)
    if xmin is not None: plt.xlim(xmin=xmin)
    if xmax is not None: plt.xlim(xmax=xmax)
    if ymin is not None: plt.ylim(ymin=ymin)
    if ymax is not None: plt.ylim(ymax=ymax)
    plt.title('$q = %.1f$' % q)
  plt.savefig(savefig)
  #}}}


######################################################
# Function to get the bias in the recovered median 
# value for different parameters
######################################################

def get_bias(\
  basedir='/home/prayush/projects/nsbh/TidalParameterEstimation/ParameterBiasVsSnr/SEOBNRv2/set005/TN/',\
  simdir='TN_q2.00_mNS1.35_chiBH0.50_Lambda500.0_SNR60.0/NW100_NS6000/',\
  M_inj = 3*1.35, eta_inj = 2./9., chi1_inj=0, chi2_inj=0.5, SNR_inj = 60):
    """
    Load samples for a given physical configuration, 
    1. decode the location of the corresponding run. 
    2. Compute the 90% confidence interval
    3. Compute the median value of different parameters fromt the posterior.
    """
    #{{{
    test_dir = os.path.join(basedir, simdir)
    m1_inj, m2_inj = pnutils.mchirp_eta_to_mass1_mass2(M_inj * eta_inj**0.6, eta_inj)
    params_inj = {'eta' : eta_inj, 'Mtot' : M_inj, 'Mc' : M_inj * eta_inj**0.6,
                  'chi1' : chi1_inj, 'chi2' : chi2_inj, 
                  'm1' : m1_inj, 'm2' : m2_inj}
    
    match = {}
    match['samples'] = load_samples_join(test_dir, SNR_inj)
    
    bias, CI90, width90 = {}, {}, {}
    for param in ['eta', 'Mc', 'chi1', 'chi2', 'm1', 'm2', 'Mtot']:
      S = match['samples'][param]
      if S == None: return None, [None, None], None
      param_inj = params_inj[param]
      if 'chi' in param: bias[param] = (np.median(S) - param_inj) / 1.
      else: bias[param] = (np.median(S) - param_inj) / param_inj
      CI90[param] = [np.percentile(S, 5), np.percentile(S, 95)]
      if 'chi' in param: width90[param] = CI90[param][1] - CI90[param][0]
      else: width90[param] = (CI90[param][1] - CI90[param][0]) / param_inj    
    #print "bias = %f, width90 = %f" % (bias, width90)
    return bias, CI90, width90
    #}}}

# Function to get the exact parameter bias/whatever given the injection
# parameters as input
def get_results(data, q=None, chiBH=None, NSLmbd=None, SNR=None,\
        p='Mc', qnt='median-val', \
        CI=0, CILevs=[90.0, 68.26895, 95.44997, 99.73002]):
  #{{{
  '''
  Confidence interval is indexed as {0 : 90, 1 : 68%, 2 : 95%, 3 : 99.7%}
  Parameter requested in **p** has to be one of :
  * 'm1'
  * 'm2'
  * 'Mc'
  * 'Mtot'
  * 'eta'
  * 'q'
  * 'chiBH'
  * 'Lambda'

  Quantity requested in **qnt** has to be one of :
  * 'median-val': X(median)
  * 'fbias'     : (X(median) - X(injection)) / X(injection)
  * 'CIlower'   : X[10%] or 100-x%
  * 'CIhigher'  : X[90%] or x%
  * 'CIfwidth'  : (X[90%] - X[10%]) / X(injection)
  '''
  if q is None or chiBH is None or NSLmbd is None or SNR is None:
    raise IOError("Need all of (q,chi,NSL,SNR) parameters")
  l1grp, l2grp, l3grp = \
        ['q%.1f.dir' % q, 'chiBH%.2f.dir' % chiBH, 'LambdaNS%.1f.dir' % NSLmbd]
  dset = 'SNR%.1f.dat' % SNR
  if vverbose: print "Group names = ", l1grp, l2grp, l3grp, dset
  alldata = data[l1grp][l2grp][l3grp][dset].value
  
  num_of_data_fields = 5
  if 'm1' in p: pidx = 2 + 0*num_of_data_fields
  elif 'm2' in p: pidx = 2 + 1*num_of_data_fields
  elif 'Mc' in p: pidx = 2 + 2*num_of_data_fields
  elif 'Mtot' in p: pidx = 2 + 3*num_of_data_fields
  elif 'eta' in p: pidx = 2 + 4*num_of_data_fields
  elif 'q' in p: pidx = 2 + 5*num_of_data_fields
  elif 'chiBH' in p: pidx = 2 + 6*num_of_data_fields
  elif 'Lambda' in p: pidx = 2 + 7*num_of_data_fields
  
  if 'median-val' in qnt: pidx += 0
  elif 'fbias' in qnt: pidx += 1
  elif 'CIlower' in qnt: pidx += 2
  elif 'CIhigher' in qnt: pidx += 3
  elif 'CIfwidth' in qnt: pidx += 4
  
  return alldata[CI, pidx]
  #}}}

# Function to get statistical properties of recovered parameters as functions 
# of the injected SNR
def get_results_vs_snr(data, q=None, chiBH=None, NSLmbd=None,\
              p='Mc', qnt='CIfwidth', CI=0,\
              SNRvec=[20, 30, 50, 70, 90, 120]):
  #{{{
  """
  This function returns the requested quantity for the requested parameter
  as a function of the SNR from the list of input SNRs.
  
  Confidence interval is indexed as {0 : 90, 1 : 68%, 2 : 95%, 3 : 99.7%}
  Parameter requested in **p** has to be one of :
  * 'm1'
  * 'm2'
  * 'Mc'
  * 'Mtot'
  * 'eta'
  * 'chiBH'
  * 'Lambda'

  Quantity requested in **qnt** has to be one of :
  * 'median-val': X(median)
  * 'fbias'     : (X(median) - X(injection)) / X(injection)
  * 'CIlower'   : X[10%] or 100-x%
  * 'CIhigher'  : X[90%] or x%
  * 'CIfwidth'  : (X[90%] - X[10%]) / X(injection)
  """
  param_values = np.array([])
  for snr in SNRvec:
    param_values = np.append( param_values, \
        get_results(data, q=q, chiBH=chiBH, NSLmbd=NSLmbd, SNR=snr,\
                    p=p, qnt=qnt, CI=CI) )
  return np.array(SNRvec), np.array(param_values)
  #}}}

# Function to get statistical properties of recovered parameters as functions 
# of the injected LAMBDA
def get_results_vs_Lambda(data, q=None, chiBH=None, SNR=None,\
              p='Mc', qnt='CIfwidth', CI=0,\
              Lambdavec=[500, 800, 1000]):
  #{{{
  """
  This function returns the requested quantity for the requested parameter
  as a function of the Lambda from the list of input Lambdas.
  
  Confidence interval is indexed as {0 : 90, 1 : 68%, 2 : 95%, 3 : 99.7%}
  Parameter requested in **p** has to be one of :
  * 'm1'
  * 'm2'
  * 'Mc'
  * 'Mtot'
  * 'eta'
  * 'chiBH'
  * 'Lambda'

  Quantity requested in **qnt** has to be one of :
  * 'median-val': X(median)
  * 'fbias'     : (X(median) - X(injection)) / X(injection)
  * 'CIlower'   : X[10%] or 100-x%
  * 'CIhigher'  : X[90%] or x%
  * 'CIfwidth'  : (X[90%] - X[10%]) / X(injection)
  """
  param_values = np.array([])
  for Lambda in Lambdavec:
    param_values = np.append( param_values, \
        get_results(data, q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=SNR,\
                    p=p, qnt=qnt, CI=CI) )
  return np.array(Lambdavec), np.array(param_values)
  #}}}

# Function to get statistical properties of recovered parameters as functions 
# of the injected SNR
def get_results_vs_parameter(data, q=None, chiBH=None, NSLmbd=None, SNR=None,\
                        xp='SNR', xvec=None, p='Mc', qnt='CIfwidth', CI=0):
  #{{{
  """
  This function returns the requested quantity for the requested parameter
  as a function of the SNR from the list of input SNRs.
  
  Confidence interval is indexed as {0 : 90, 1 : 68%, 2 : 95%, 3 : 99.7%}
  Parameter requested in **p** has to be one of :
  * 'm1'
  * 'm2'
  * 'Mc'
  * 'Mtot'
  * 'eta'
  * 'chiBH'
  * 'Lambda'

  Quantity requested in **qnt** has to be one of :
  * 'median-val': X(median)
  * 'fbias'     : (X(median) - X(injection)) / X(injection)
  * 'CIlower'   : X[10%] or 100-x%
  * 'CIhigher'  : X[90%] or x%
  * 'CIfwidth'  : (X[90%] - X[10%]) / X(injection)
  """
  param_values = np.array([])
  for x in xvec:    
    if 'q' in xp: q = x
    elif 'chi' in xp: chiBH = x
    elif 'Lambda' in xp: NSLmbd = x
    elif 'SNR' in xp or 'snr' in xp: SNR = x
    else:
      raise IOError("Parameter %s not supported by get_results_vs_parameter"%xp)
    #
    param_values = np.append( param_values, \
                  get_results(data, q=q, chiBH=chiBH, NSLmbd=NSLmbd, SNR=SNR,\
                              p=p, qnt=qnt, CI=CI) )
  #
  return np.array(xvec), np.array(param_values)
  #}}}


# Function to find out when a given statistical property of a recovered 
# parameter attains a target value
def get_snr_where_quantity_val_reached(data, q=None, chiBH=None, NSLmbd=None,\
              p='Lambda', qnt='CIfwidth', CI=0, target_val=1., \
              SNRvec=[20, 30, 50, 70, 90, 120]):
  #{{{
  """
  This function :
  - Calculates the requested quantitiy as a function of injected SNR
  - Calculates the SNR threshold where this quantity attains the required value
  - Returns this SNR
  
  Confidence interval is indexed as {0 : 90, 1 : 68%, 2 : 95%, 3 : 99.7%}
  Parameter requested in **p** has to be one of :
  * 'm1'
  * 'm2'
  * 'Mc'
  * 'Mtot'
  * 'eta'
  * 'chiBH'
  * 'Lambda'

  Quantity requested in **qnt** has to be one of :
  * 'median-val': X(median)
  * 'fbias'     : (X(median) - X(injection)) / X(injection)
  * 'CIlower'   : X[10%] or 100-x%
  * 'CIhigher'  : X[90%] or x%
  * 'CIfwidth'  : (X[90%] - X[10%]) / X(injection)
  
  Desired level of the value is given in **target_value**
  """
  snr, values = get_results_vs_snr(data, q=q, chiBH=chiBH, NSLmbd=NSLmbd,\
                    p=p, qnt=qnt, CI=CI, SNRvec=SNRvec)
  valuesI = UnivariateSpline(snr, values)
  
  def fun(snr): return np.abs(valuesI(snr) - target_val)
  result = minimize_scalar(fun, bounds=[snr.min(), snr.max()], method='Bounded')
  
  return result['x'], result['fun']
  #}}}

def get_Lambda_where_quantity_val_reached(data, q=None, chiBH=None, SNR=None,\
        p='Lambda', qnt='CIfwidth', CI=0, target_val=1., \
        Lambdavec=[500,800,1000]):
  #{{{
  """
  This function :
  - Calculates the requested quantitiy as a function of injected NS Lambda
  - Calculates the Lambda threshold where this quantity attains the required value
  - Returns this Lambda
  
  Confidence interval is indexed as {0 : 90, 1 : 68%, 2 : 95%, 3 : 99.7%}
  Parameter requested in **p** has to be one of :
  * 'm1'
  * 'm2'
  * 'Mc'
  * 'Mtot'
  * 'eta'
  * 'chiBH'
  * 'Lambda'

  Quantity requested in **qnt** has to be one of :
  * 'median-val': X(median)
  * 'fbias'     : (X(median) - X(injection)) / X(injection)
  * 'CIlower'   : X[10%] or 100-x%
  * 'CIhigher'  : X[90%] or x%
  * 'CIfwidth'  : (X[90%] - X[10%]) / X(injection)
  
  Desired level of the value is given in **target_value**
  """
  Lambda, values = get_results_vs_Lambda(data, q=q, chiBH=chiBH, SNR=SNR,\
                      p=p, qnt=qnt, CI=CI, Lambdavec=Lambdavec)
  valuesI = UnivariateSpline(Lambda, values, k=np.min([3, len(values)-1]))
  
  def fun(Lambda): return np.abs( valuesI(Lambda) - target_val )
  result = minimize_scalar(fun, bounds=[Lambda.min(), Lambda.max()], method='Bounded')
  
  return result['x'], result['fun']
  #}}}


# Function to find out when a given statistical property of a recovered 
# parameter attains a target value
def get_parameter_where_quantity_val_reached(data, q=None, chiBH=None, \
          NSLmbd=None, SNR=None, \
          xp='SNR', xvec=None, p='Lambda', qnt='CIfwidth', CI=0, target_val=1.):
  #{{{
  """
  This function :
  - Calculates the requested quantitiy as a function of injected parameter
  - Calculates the parameter threshold where this quantity attains the required value
  - Returns this parameter value
  
  Confidence interval is indexed as {0 : 90, 1 : 68%, 2 : 95%, 3 : 99.7%}
  Parameter requested in **p** has to be one of :
  * 'm1'
  * 'm2'
  * 'Mc'
  * 'Mtot'
  * 'eta'
  * 'chiBH'
  * 'Lambda'

  Quantity requested in **qnt** has to be one of :
  * 'median-val': X(median)
  * 'fbias'     : (X(median) - X(injection)) / X(injection)
  * 'CIlower'   : X[10%] or 100-x%
  * 'CIhigher'  : X[90%] or x%
  * 'CIfwidth'  : (X[90%] - X[10%]) / X(injection)
  
  Desired level of the value is given in **target_value**
  """
  params, values = get_results_vs_parameter(data, q=q, chiBH=chiBH,\
                          NSLmbd=NSLmbd, SNR=SNR,\
                          xp=xp, xvec=xvec, p=p, qnt=qnt, CI=CI)
  valuesI = UnivariateSpline(params, values)
  #valuesI = interp1d(params, values)
  
  def fun(x): return np.abs(valuesI(x) - target_val)
  result = minimize_scalar(fun, bounds=[params.min(), params.max()], method='Bounded')
  
  return result['x'], result['fun']
  #}}}



######################################################
######################################################
# BEGIN
######################################################
######################################################

######################################################
# INPUTS
######################################################
# Injection / Recovery information
inject_tidal = recover_tidal = False
if len(sys.argv) >= 3:
  if int(sys.argv[1]) != 0: inject_tidal = True
  if int(sys.argv[2]) != 0: recover_tidal = True

# plotting flags
plot_bias = plot_width = False
if len(sys.argv) >= 5:
  if int(sys.argv[3]) != 0: plot_bias = True
  if int(sys.argv[4]) != 0: plot_width = True


######################################################
# Set up parameters of signal
######################################################
chi1 = 0.                           # small BH
chi2vec = np.array([-0.5, 0, 0.5, 0.74999])  # larger BH
mNS = 1.35
qvec = np.array([2, 3, 4, 5])
#qvec = np.array([2, 3, 4])
etavec = qvec / (1. + qvec)**2
mtotalvec = mNS + mNS * qvec
mchirpvec = mtotalvec * etavec**0.6
Lambdavec = np.array([0])#[500, 800, 1000])
SNRvec = np.array([20, 30, 50, 70, 90, 120])
if inject_tidal: Lambdavec = np.array([500, 800, 1000, 1500, 2000])
#if inject_tidal: Lambdavec = np.array([500, 800, 1000])

######################################################
# Set up parameters of templates
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
simstring = sigstring + tmpstring

Nwalkers = [100]
Nsamples = [150000]
Nburnin  = 500

plotdir = 'plots' + simstring + '/'
figtype = 'png'

######################################################
# Read in parameter biases and other data
######################################################
datadir  = '/home/prayush/research/NSBH/TidalParameterEstimation/FinalCombinedPEData/'
#datadir  = '/home/prayush/research/NSBH/TidalParameterEstimation/ParameterBiasVsSnr/SEOBNRv2/set005/' + simstring
datafile = simstring + '_ParameterBiasesAndConfidenceIntervals.h5'
datafile = os.path.join(datadir, datafile)

if not recover_tidal:
  datafileNN = os.path.join('/home/prayush/research/NSBH/TidalParameterEstimation/ParameterBiasVsSnr/SEOBNRv2/set005/NN','NN_ParameterBiasesAndConfidenceIntervals.h5')
  #dataNN = h5py.File(datafileNN, 'r')
  #datafile = os.path.join('/home/prayush/research/NSBH/TidalParameterEstimation/ParameterBiasVsSnr/SEOBNRv2/set005/TN','TN_ParameterBiasesAndConfidenceIntervals.h5')
#else:
#  datafile = os.path.join('/home/prayush/research/NSBH/TidalParameterEstimation/ParameterBiasVsSnr/SEOBNRv2/set005/TT','TT_ParameterBiasesAndConfidenceIntervals.h5')
  
print '''Reading from data file : %s''' % datafile

data = h5py.File(datafile, 'r')

print \
'''
The data stored in the HDF file has the following format:
- Level1 groups : Mass-ratio q. name-format is ('q%.1f.dir' % q)
- Level2 groups : Black-hole spins. name-format is ('chiBH%.2f.dir' % chiBH)
- Level3 groups : Neutron star Lambda. name-format is ('LambdaNS%.1f.dir' % LambdaNS)
- Dataset : Injection's SNR. name-format is ('SNR%.1f.dat' % SNR)
'''
print \
'''
The dataset is a 2-d array, with rows corresponding to confidence-levels. 
We use [90%, 1sigma, 2sigma, 3sigma] confidence probabilities, and have the 
confidence interval associated with the posterior of each of the binary's 
physical parameters, i.e. ['m1', 'm2', 'Mc', 'Mtot', 'eta', 'chi2', 'Lambda']

We have stored 5 quantities for each parameter k:
k+0: Median recovered parameter value from posterior samples
k+1: Bias = (X(median) - X(inj))/X(inj)
k+2: Confidence level lower bound = X(confidence_level_low)
k+3: Confidence level upper bound = X(confidence_level_up)
k+4: Confidence interval = (X(confidence_level_up) - X(confidence_level_low))/X(inj)
    
PLUS, the first 2 columns in each row of the dataset are:
0 : Confidence interval probability (actual number \in [0,1])
1 : max LogLikelihood value (actually the match value, which can be 
    converted to LogLikelihood as LogL = 1/2 SNR^2 MATCH^2
'''

######################################################
# PLOTTING PARAMETERS
######################################################

# Plots showing Mass/Spin/Lambda recovery biases
plot_MchirpBias = False
plot_EtaBias    = False
plot_QBias      = False
plot_sBHBias    = False
plot_mBHBias    = False
plot_LambdaBias = False

# Plots showing Lambda recovery
plot_SNRcrit    = False
plot_LambdaCrit = False
plot_LambdaRecoveryCurves = True
plot_EtaCriticalCurves    = True

# Reduced ranges for plots
plotSNRvec    = np.array([20, 30, 50, 70])
plotqvec      = qvec
plotchi2vec   = np.array([0, 0.5, 0.74999])
plotLambdavec = np.array([500, 1000, 1500])

# Definition of "measurability" for Lambda
error_threshold = 1.
error_p         = error_threshold * 100.

######################################################
######################################################
# Time to MAKE PLOTS
######################################################
######################################################
print "MAKING PLOTS NOW.."

######################################################
print "\n\n\n MAKING Plots with/for CHIRP MASS "
######################################################
if plot_MchirpBias:
  print \
"""
Plotting the fractional bias in chirp-mass recovery at different SNR levels, as 
a function of the BH mass and spin. In addition also plotting the width of the
confidence interval for chirp-mass, i.e the statistical uncertainty around the
maximum likelihood value.

The bias shown is a fraction of the injected Lambda, and different figures are
made for different LAmbda values
"""
  mchirpBiases, mchirpCIwidths = {}, {}
  for Lambda in Lambdavec:
    mchirpBiases[Lambda], mchirpCIwidths[Lambda] = {}, {}
    for snr in SNRvec:
      mchirpBiases[Lambda][snr], mchirpCIwidths[Lambda][snr] = {}, {}
      for CI in range( len(CILevels) ):
        mchirpBiases[Lambda][snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
        mchirpCIwidths[Lambda][snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
        for i, q in enumerate(qvec):
          for j, chiBH in enumerate(chi2vec):
            if vverbose:
              print "getting bias in mchirp for q=%f, chiBh=%f, Lambda=%f at SNR = %f" %\
                    (q, chiBH, Lambda, snr)
            try:
              mchirpBiases[Lambda][snr][CI][i,j] = get_results(data,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='Mc', qnt='fbias', CI=CI)
            except KeyError as kerr:
              print kerr
              mchirpBiases[Lambda][snr][CI][i,j] = get_results(dataNN,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='Mc', qnt='fbias', CI=CI)
              
            try:
              mchirpCIwidths[Lambda][snr][CI][i,j] = get_results(data,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='Mc', qnt='CIfwidth', CI=CI)
            except KeyError:
              mchirpCIwidths[Lambda][snr][CI][i,j] = get_results(dataNN,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='Mc', qnt='CIfwidth', CI=CI)
  #
  for plotCI in [0,1,2,3]:
    # First we want to plot the width of confidence intervals in the posterior
    #    including the Lambda=0, i.e. BHBH inj case
    if vverbose: print "Making CHIRP MASS plots for CI = %f" % CILevels[plotCI]
    Xarray, Yarray, Zarray1, Zarray2 = [], [], [], []
    titles = []
    
    for Lambda in Lambdavec:
      Xtmp, Ytmp, Ztmp1, Ztmp2 = [], [], [], []
      ttmp = []
      for snr in plotSNRvec:
        Xtmp.append(np.array(chi2vec))
        Ytmp.append(mNS * np.array(qvec))
        Ztmp1.append(mchirpBiases[Lambda][snr][plotCI] * 100)
        Ztmp2.append(mchirpCIwidths[Lambda][snr][plotCI] * 100)
        ttmp.append('$\Lambda_\mathrm{NS}=%.1f, \\rho=%.1f$' % (Lambda, snr))
      Xarray.append(Xtmp)
      Yarray.append(Ytmp)
      Zarray1.append(Ztmp1)
      Zarray2.append(Ztmp2)
      titles.append(ttmp)
      
    make_contour_array(Xarray, Yarray, Zarray2, \
      xlabel='Black-hole spin', ylabel='$M_\mathrm{BH}(M_\odot)$', cmap=cm.Reds_r,\
      xmin=min(chi2vec), xmax=max(chi2vec), ymin=min(qvec)*mNS, ymax=max(qvec)*mNS, titles=titles, \
      clabel="$(\Delta\mathcal{M}_c)^{%.1f \%%}/\mathcal{M}_c^\mathrm{Injected}\\times 100$" % CILevels[plotCI], \
      levelspacing=0.002*1.5, cfmt='%.2f',\
      figname=os.path.join(plotdir,simstring+('MchirpCIWidths%.1f_Lambda_SNR' % CILevels[plotCI]).replace('.','_')+'.'+figtype))
  
    Xarray, Yarray, Zarray1, Zarray2 = [], [], [], []
    titles = []
    
    # Now we want to plot the systematic bias of the median value from posterior
    #  without including the Lambda=0 case, as templates should exactly match it
    for Lambda in Lambdavec[Lambdavec != 0]:
      Xtmp, Ytmp, Ztmp1, Ztmp2 = [], [], [], []
      ttmp = []
      for snr in plotSNRvec:
        Xtmp.append(np.array(chi2vec))
        Ytmp.append(mNS * np.array(qvec))
        Ztmp1.append(mchirpBiases[Lambda][snr][plotCI] * 100)
        Ztmp2.append(mchirpCIwidths[Lambda][snr][plotCI] * 100)
        ttmp.append('$\Lambda_\mathrm{NS}=%.1f, \\rho=%.1f$' % (Lambda, snr))
      Xarray.append(Xtmp)
      Yarray.append(Ytmp)
      Zarray1.append(Ztmp1)
      Zarray2.append(Ztmp2)
      titles.append(ttmp)

    make_contour_array(Xarray, Yarray, Zarray1, \
      xlabel='Black-hole spin', ylabel='$M_\mathrm{BH}(M_\odot)$', \
      xmin=min(chi2vec), xmax=max(chi2vec), ymin=min(qvec)*mNS, ymax=max(qvec)*mNS, titles=titles, cmap=cm.rainbow,\
      clabel='$100\\times (\mathcal{M}_c^\mathrm{Median}-\mathcal{M}_c^\mathrm{Injected})/\mathcal{M}_c^\mathrm{Injected}$', \
      levelspacing=1.e-3,  cfmt='%.3f',\
      figname=os.path.join(plotdir,simstring+('MchirpBiases_CI%.1f_Lambda_SNR' % CILevels[plotCI]).replace('.','_')+'.'+figtype))


######################################################
print "\n\n\n MAKING Plots with/for MASS RATIO "
######################################################
if plot_EtaBias:
  print \
"""
Plotting the fractional bias in MASS-RATIO recovery at different SNR levels, as 
a function of the BH mass and spin. In addition also plotting the width of the
confidence interval for chirp-mass, i.e the statistical uncertainty around the
maximum likelihood value.

The bias shown is a fraction of the injected Lambda, and different figures are
made for different LAmbda values
"""
  etaBiases, etaCIwidths = {}, {}
  for Lambda in Lambdavec:
    etaBiases[Lambda], etaCIwidths[Lambda] = {}, {}
    for snr in SNRvec:
      etaBiases[Lambda][snr], etaCIwidths[Lambda][snr] = {}, {}
      for CI in range( len(CILevels) ):
        etaBiases[Lambda][snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
        etaCIwidths[Lambda][snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
        for i, q in enumerate(qvec):
          for j, chiBH in enumerate(chi2vec):
            if vverbose:
              print "getting bias in eta for q=%f, chiBh=%f, Lambda=%f at SNR = %f" %\
                    (q, chiBH, Lambda, snr)
            try:
              etaBiases[Lambda][snr][CI][i,j] = get_results(data,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='eta', qnt='fbias', CI=CI)
            except KeyError:
              etaBiases[Lambda][snr][CI][i,j] = get_results(dataNN,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='eta', qnt='fbias', CI=CI)
            try:
              etaCIwidths[Lambda][snr][CI][i,j] = get_results(data,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='eta', qnt='CIfwidth', CI=CI)
            except KeyError:
              etaCIwidths[Lambda][snr][CI][i,j] = get_results(dataNN,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='eta', qnt='CIfwidth', CI=CI)
  #
  for plotCI in [0,1,2,3]:
    # First we want to plot the width of confidence intervals in the posterior
    #    including the Lambda=0, i.e. BHBH inj case
    if vverbose: print "Making ETA plots for CI = %f" % CILevels[plotCI]
    
    # MAKE ETA CONFIDENCE INTERVAL PLOTS
    Xarray, Yarray, Zarray2 = [], [], []
    titles = []
    
    for Lambda in Lambdavec:
      Xtmp, Ytmp, Ztmp2 = [], [], []
      ttmp = []
      for snr in plotSNRvec:
        Xtmp.append(np.array(chi2vec))
        Ytmp.append(mNS * np.array(qvec))
        Ztmp2.append(etaCIwidths[Lambda][snr][plotCI] * 100)
        ttmp.append('$\Lambda_\mathrm{NS}=%.1f, \\rho=%.1f$' % (Lambda, snr))
      Xarray.append(Xtmp)
      Yarray.append(Ytmp)
      Zarray2.append(Ztmp2)
      titles.append(ttmp)
      
    make_contour_array(Xarray, Yarray, Zarray2, \
      xlabel='Black-hole spin', ylabel='$M_\mathrm{BH}(M_\odot)$', cmap=cm.Reds_r,\
      xmin=min(chi2vec), xmax=max(chi2vec), ymin=min(qvec)*mNS, ymax=max(qvec)*mNS, titles=titles, \
      clabel="$(\Delta\eta)^{%.1f \%%}/\eta^\mathrm{Injected}\\times 100$" % CILevels[plotCI], \
      levelspacing=0.5, cfmt='%.0f',\
      figname=os.path.join(plotdir,simstring+('EtaCIWidths%.1f_Lambda_SNR' % CILevels[plotCI]).replace('.','_')+'.'+figtype))
    
    # MAKE ETA BIAS PLOTS
    Xarray, Yarray, Zarray = [], [], []
    titles = []
    
    # Now we want to plot the systematic bias of the median value from posterior
    #  without including the Lambda=0 case, as templates should exactly match it
    for Lambda in Lambdavec[Lambdavec != 0]:
      Xtmp, Ytmp, Ztmp = [], [], []
      ttmp = []
      for snr in plotSNRvec:
        Xtmp.append(np.array(chi2vec))
        Ytmp.append(mNS * np.array(qvec))
        Ztmp.append(etaBiases[Lambda][snr][plotCI] * 100)
        ttmp.append('$\Lambda_\mathrm{NS}=%.1f, \\rho=%.1f$' % (Lambda, snr))
      Xarray.append(Xtmp)
      Yarray.append(Ytmp)
      Zarray.append(Ztmp)
      titles.append(ttmp)

    make_contour_array(Xarray, Yarray, Zarray, \
      xlabel='Black-hole spin', ylabel='$M_\mathrm{BH}(M_\odot)$', \
      xmin=min(chi2vec), xmax=max(chi2vec), ymin=min(qvec)*mNS, ymax=max(qvec)*mNS, titles=titles, cmap=cm.rainbow_r,\
      clabel='$100\\times (\eta^\mathrm{Median}-\eta^\mathrm{Injected})/\mathcal{M}_c^\mathrm{Injected}$', \
      levelspacing=0.4,  cfmt='%.1f',\
      figname=os.path.join(plotdir,simstring+('EtaBiases_CI%.1f_Lambda_SNR' % CILevels[plotCI]).replace('.','_')+'.'+figtype))

if plot_QBias:
  print \
"""
Plotting the fractional bias in MASS-RATIO recovery at different SNR levels, as 
a function of the BH mass and spin. In addition also plotting the width of the
confidence interval for chirp-mass, i.e the statistical uncertainty around the
maximum likelihood value.

The bias shown is a fraction of the injected Lambda, and different figures are
made for different LAmbda values
"""
  etaBiases, etaCIwidths = {}, {}
  for Lambda in Lambdavec:
    etaBiases[Lambda], etaCIwidths[Lambda] = {}, {}
    for snr in SNRvec:
      etaBiases[Lambda][snr], etaCIwidths[Lambda][snr] = {}, {}
      for CI in range( len(CILevels) ):
        etaBiases[Lambda][snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
        etaCIwidths[Lambda][snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
        for i, q in enumerate(qvec):
          for j, chiBH in enumerate(chi2vec):
            if vverbose:
              print "getting bias in eta for q=%f, chiBh=%f, Lambda=%f at SNR = %f" %\
                    (q, chiBH, Lambda, snr)
            try:
              etaBiases[Lambda][snr][CI][i,j] = get_results(data,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='eta', qnt='fbias', CI=CI)
            except KeyError:
              etaBiases[Lambda][snr][CI][i,j] = get_results(dataNN,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='eta', qnt='fbias', CI=CI)
            try:
              etaCIwidths[Lambda][snr][CI][i,j] = get_results(data,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='eta', qnt='CIfwidth', CI=CI)
            except KeyError:
              etaCIwidths[Lambda][snr][CI][i,j] = get_results(dataNN,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='eta', qnt='CIfwidth', CI=CI)
  #
  for plotCI in [0,1,2,3]:
    # First we want to plot the width of confidence intervals in the posterior
    #    including the Lambda=0, i.e. BHBH inj case
    if vverbose: print "Making ETA plots for CI = %f" % CILevels[plotCI]
    
    # MAKE ETA CONFIDENCE INTERVAL PLOTS
    Xarray, Yarray, Zarray2 = [], [], []
    titles = []
    
    for Lambda in Lambdavec:
      Xtmp, Ytmp, Ztmp2 = [], [], []
      ttmp = []
      for snr in plotSNRvec:
        Xtmp.append(np.array(chi2vec))
        Ytmp.append(mNS * np.array(qvec))
        Ztmp2.append(etaCIwidths[Lambda][snr][plotCI] * 100)
        ttmp.append('$\Lambda_\mathrm{NS}=%.1f, \\rho=%.1f$' % (Lambda, snr))
      Xarray.append(Xtmp)
      Yarray.append(Ytmp)
      Zarray2.append(Ztmp2)
      titles.append(ttmp)
      
    make_contour_array(Xarray, Yarray, Zarray2, \
      xlabel='Black-hole spin', ylabel='$M_\mathrm{BH}(M_\odot)$', cmap=cm.Reds_r,\
      xmin=min(chi2vec), xmax=max(chi2vec), ymin=min(qvec)*mNS, ymax=max(qvec)*mNS, titles=titles, \
      clabel="$(\Delta\eta)^{%.1f \%%}/\eta^\mathrm{Injected}\\times 100$" % CILevels[plotCI], \
      levelspacing=0.5, cfmt='%.0f',\
      figname=os.path.join(plotdir,simstring+('EtaCIWidths%.1f_Lambda_SNR' % CILevels[plotCI]).replace('.','_')+'.'+figtype))
    
    # MAKE ETA BIAS PLOTS
    Xarray, Yarray, Zarray = [], [], []
    titles = []
    
    # Now we want to plot the systematic bias of the median value from posterior
    #  without including the Lambda=0 case, as templates should exactly match it
    for Lambda in Lambdavec[Lambdavec != 0]:
      Xtmp, Ytmp, Ztmp = [], [], []
      ttmp = []
      for snr in plotSNRvec:
        Xtmp.append(np.array(chi2vec))
        Ytmp.append(mNS * np.array(qvec))
        Ztmp.append(etaBiases[Lambda][snr][plotCI] * 100)
        ttmp.append('$\Lambda_\mathrm{NS}=%.1f, \\rho=%.1f$' % (Lambda, snr))
      Xarray.append(Xtmp)
      Yarray.append(Ytmp)
      Zarray.append(Ztmp)
      titles.append(ttmp)

    make_contour_array(Xarray, Yarray, Zarray, \
      xlabel='Black-hole spin', ylabel='$M_\mathrm{BH}(M_\odot)$', \
      xmin=min(chi2vec), xmax=max(chi2vec), ymin=min(qvec)*mNS, ymax=max(qvec)*mNS, titles=titles, cmap=cm.rainbow_r,\
      clabel='$100\\times (\eta^\mathrm{Median}-\eta^\mathrm{Injected})/\mathcal{M}_c^\mathrm{Injected}$', \
      levelspacing=0.4,  cfmt='%.1f',\
      figname=os.path.join(plotdir,simstring+('EtaBiases_CI%.1f_Lambda_SNR' % CILevels[plotCI]).replace('.','_')+'.'+figtype))


######################################################
print "\n\n\n MAKING Plots with/for BLACK-HOLE SPIN "
######################################################
if plot_sBHBias:
  print \
"""
Plotting the fractional bias in BH spin recovery at different SNR levels, as 
a function of the BH mass and spin. In addition also plotting the width of the
confidence interval for chirp-mass, i.e the statistical uncertainty around the
maximum likelihood value.

The bias shown is a fraction of the injected Lambda, and different figures are
made for different LAmbda values
"""
  chiBHBiases, chiBHCIwidths = {}, {}
  for Lambda in Lambdavec:
    chiBHBiases[Lambda], chiBHCIwidths[Lambda] = {}, {}
    for snr in SNRvec:
      chiBHBiases[Lambda][snr], chiBHCIwidths[Lambda][snr] = {}, {}
      for CI in range( len(CILevels) ):
        chiBHBiases[Lambda][snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
        chiBHCIwidths[Lambda][snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
        for i, q in enumerate(qvec):
          for j, chiBH in enumerate(chi2vec):
            if vverbose:
              print "getting bias in chiBH for q=%f, chiBh=%f, Lambda=%f at SNR = %f" %\
                    (q, chiBH, Lambda, snr)
            try:
              chiBHBiases[Lambda][snr][CI][i,j] = get_results(data,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='chiBH', qnt='fbias', CI=CI)
            except KeyError:
              chiBHBiases[Lambda][snr][CI][i,j] = get_results(dataNN,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='chiBH', qnt='fbias', CI=CI)
            try:
              chiBHCIwidths[Lambda][snr][CI][i,j] = get_results(data,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='chiBH', qnt='CIfwidth', CI=CI)
            except KeyError:
              chiBHCIwidths[Lambda][snr][CI][i,j] = get_results(dataNN,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='chiBH', qnt='CIfwidth', CI=CI)
  #
  for plotCI in [0,1,2,3]:
    # First we want to plot the width of confidence intervals in the posterior
    #    including the Lambda=0, i.e. BHBH inj case
    if vverbose: print "Making chiBH plots for CI = %f" % CILevels[plotCI]
    
    # MAKE chiBH CONFIDENCE INTERVAL PLOTS
    Xarray, Yarray, Zarray2 = [], [], []
    titles = []
    
    for Lambda in Lambdavec:
      Xtmp, Ytmp, Ztmp2 = [], [], []
      ttmp = []
      for snr in plotSNRvec:
        Xtmp.append(np.array(chi2vec))
        Ytmp.append(mNS * np.array(qvec))
        Ztmp2.append(chiBHCIwidths[Lambda][snr][plotCI] * 1)
        ttmp.append('$\Lambda_\mathrm{NS}=%.1f, \\rho=%.1f$' % (Lambda, snr))
      Xarray.append(Xtmp)
      Yarray.append(Ytmp)
      Zarray2.append(Ztmp2)
      titles.append(ttmp)
      
    make_contour_array(Xarray, Yarray, Zarray2, \
      xlabel='Black-hole spin', ylabel='$M_\mathrm{BH}(M_\odot)$', cmap=cm.Reds_r,\
      xmin=min(chi2vec), xmax=max(chi2vec), ymin=min(qvec)*mNS, ymax=max(qvec)*mNS, titles=titles, \
      clabel="$(\Delta\chi_\mathrm{BH})^{%.1f \%%}$" % CILevels[plotCI], \
      levelspacing=0.01, cfmt='%.2f',\
      figname=os.path.join(plotdir,simstring+('ChiBHCIWidths%.1f_Lambda_SNR' % CILevels[plotCI]).replace('.','_')+'.'+figtype))
    
    # MAKE chiBH BIAS PLOTS
    Xarray, Yarray, Zarray = [], [], []
    titles = []
    
    # Now we want to plot the systematic bias of the median value from posterior
    #  without including the Lambda=0 case, as templates should exactly match it
    for Lambda in Lambdavec[Lambdavec != 0]:
      Xtmp, Ytmp, Ztmp = [], [], []
      ttmp = []
      for snr in plotSNRvec:
        Xtmp.append(np.array(chi2vec))
        Ytmp.append(mNS * np.array(qvec))
        Ztmp.append(chiBHBiases[Lambda][snr][plotCI] * 1)
        ttmp.append('$\Lambda_\mathrm{NS}=%.1f, \\rho=%.1f$' % (Lambda, snr))
      Xarray.append(Xtmp)
      Yarray.append(Ytmp)
      Zarray.append(Ztmp)
      titles.append(ttmp)

    make_contour_array(Xarray, Yarray, Zarray, \
      xlabel='Black-hole spin', ylabel='$M_\mathrm{BH}(M_\odot)$', \
      xmin=min(chi2vec), xmax=max(chi2vec), ymin=min(qvec)*mNS, ymax=max(qvec)*mNS, titles=titles, cmap=cm.rainbow_r,\
      clabel='$\chi_\mathrm{BH}^\mathrm{Median}-\chi_\mathrm{BH}^\mathrm{Injected}$', \
      levelspacing=0.005,  cfmt='%.2f',\
      figname=os.path.join(plotdir,simstring+('ChiBHBiases_CI%.1f_Lambda_SNR' % CILevels[plotCI]).replace('.','_')+'.'+figtype))


######################################################
print "\n\n\n MAKING Plots with/for NS LAMBDA "
######################################################

if plot_LambdaBias and recover_tidal:
  print \
"""
Plotting the fractional bias in Lambda recovery at different SNR levels, as a 
function of the BH mass and spin. 

The bias shown is a fraction of the injected Lambda, and different figures are
made for different LAmbda values
"""
  lambdaBiases, lambdaCIwidths = {}, {}
  for Lambda in Lambdavec:
    lambdaBiases[Lambda], lambdaCIwidths[Lambda] = {}, {}
    for snr in SNRvec:
      lambdaBiases[Lambda][snr], lambdaCIwidths[Lambda][snr] = {}, {}
      for CI in range( len(CILevels) ):
        lambdaBiases[Lambda][snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
        lambdaCIwidths[Lambda][snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
        for i, q in enumerate(qvec):
          for j, chiBH in enumerate(chi2vec):
            if vverbose:
              print "getting bias in Lambda for q=%f, chiBh=%f, Lambda=%f at SNR = %f" %\
                    (q, chiBH, Lambda, snr)
            lambdaBiases[Lambda][snr][CI][i,j] = get_results(data,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='Lambda', qnt='fbias', CI=CI)
            lambdaCIwidths[Lambda][snr][CI][i,j] = get_results(data,\
                    q=q, chiBH=chiBH, NSLmbd=Lambda, SNR=snr, \
                    p='Lambda', qnt='CIfwidth', CI=CI)
  #
  plotCI = 0
  for plotCI in [0,1,2,3]:
    Xarray, Yarray, Zarray1, Zarray2 = [], [], [], []
    titles = []
    
    for Lambda in Lambdavec:
      Xtmp, Ytmp, Ztmp1, Ztmp2 = [], [], [], []
      ttmp = []
      for snr in plotSNRvec:
        Xtmp.append(np.array(chi2vec))
        Ytmp.append(mNS * np.array(qvec))
        Ztmp1.append(lambdaBiases[Lambda][snr][plotCI] * 100)
        Ztmp2.append(lambdaCIwidths[Lambda][snr][plotCI] * 100)
        ttmp.append('$\Lambda_\mathrm{NS}=%.1f, \\rho=%.1f$' % (Lambda, snr))
      Xarray.append(Xtmp)
      Yarray.append(Ytmp)
      Zarray1.append(Ztmp1)
      Zarray2.append(Ztmp2)
      titles.append(ttmp)
      
    make_contour_array(Xarray, Yarray, Zarray2, \
      xlabel='Black-hole spin', ylabel='$M_\mathrm{BH}(M_\odot)$', cmap=cm.Spectral_r,\
      xmin=min(chi2vec), xmax=max(chi2vec), ymin=min(qvec)*mNS, ymax=max(qvec)*mNS, titles=titles, \
      clabel="$(\Delta\Lambda_\mathrm{NS})^{%.1f \%%}/\Lambda_\mathrm{NS}^\mathrm{Injected}\\times 100$" % CILevels[plotCI], levelspacing=0.5, \
      figname=os.path.join(plotdir,simstring+('LambdaCIWidths%.1f_Lambda_SNR' % CILevels[plotCI]).replace('.','_')+'.'+figtype))
  
    make_contour_array(Xarray, Yarray, Zarray1, \
      xlabel='Black-hole spin', ylabel='$M_\mathrm{BH}(M_\odot)$', \
      xmin=min(chi2vec), xmax=max(chi2vec), ymin=min(qvec)*mNS, ymax=max(qvec)*mNS, titles=titles, cmap=cm.rainbow,\
      clabel='$100\\times (\Lambda_\mathrm{NS}^\mathrm{Median}-\Lambda_\mathrm{NS}^\mathrm{Injected})/\Lambda_\mathrm{NS}^\mathrm{Injected}$', levelspacing=0.5, \
      figname=os.path.join(plotdir,simstring+('LambdaBiases_CI%.1f_Lambda_SNR' % CILevels[plotCI]).replace('.','_')+'.'+figtype))



if plot_SNRcrit and recover_tidal:
  print \
"""
Plotting the SNR threshold below which our measurement error on the NS Lambda
parameter are 100%, i.e. the SRN below which we cannot make statements about
the tidal deformability of the Neutron Star
"""
  #
  snrThresholds = {}
  #
  for Lambda in Lambdavec:
    snrThresholds[Lambda] = {}
    for CI in range( len(CILevels) ):
      snrThresholds[Lambda][CI] = np.zeros( (len(qvec), len(chi2vec)) )
      for i, q in enumerate(qvec):
        for j, chiBH in enumerate(chi2vec):
          if vverbose:
            print "getting SNR thresholds for q=%f, chiBh=%f, Lambda=%f" % \
                        (q, chiBH, Lambda)
          snrThresholds[Lambda][CI][i,j], fn =get_snr_where_quantity_val_reached(\
              data, q=q, chiBH=chiBH, NSLmbd=Lambda, \
              p='Lambda', qnt='CIfwidth', CI=CI, target_val=error_threshold,\
              SNRvec=SNRvec)
          if fn > 1.e-2:
            print "\n\n warning : Lambda only bound to %f %%" % (100*(fn+1))
            print "\t for q=%f, chiBh=%f, Lambda=%f" % (q, chiBH, Lambda)
            snrThresholds[Lambda][CI][i,j] = np.max(SNRvec)
  #
  for Lambda in Lambdavec:
    for CI in range( len(CILevels) ):
      if verbose:
        print "Plotting for Lambda injected = %f, at Confidence level = %f" % \
                (Lambda, CILevels[CI])
      
      snrThresh = snrThresholds[Lambda][CI]
      make_contour(chi2vec, qvec, snrThresh,\
        xlabel='Black-hole spin', ylabel='Binary mass-ratio',\
        clabel='SNR below which $\delta\Lambda^{%.1f \%%}_\mathrm{NS}\sim %d\%%$' % (CILevels[CI], int(error_p)),\
        title='$\Lambda_\mathrm{NS}^\mathrm{Injected}=%.1f$' %\
          (Lambda),\
        levelspacing=.5, cbfmt='%.0f', cmap=cm.autumn,\
        vmin=snrThresh.min(), vmax=snrThresh.max(), xmin=-0.5, xmax=0.75,\
        figname=os.path.join(plotdir, simstring+\
      ('SNRThresholdFor%dLambdaMeasurement_BHspin_MassRatio_Lambda%.1f_CI%.1f' %\
          (int(error_p), Lambda, CILevels[CI])).replace('.','_')+'.'+figtype))
      #
      make_contour(chi2vec, np.array(qvec) * mNS, snrThresh,\
        xlabel='Black-hole spin', ylabel='Black-hole mass $(M_\odot)$',\
        clabel='SNR below which $\delta\Lambda^{%.1f \%%}_\mathrm{NS}\sim %d\%%$' % (CILevels[CI], int(error_p)),\
        title='$\Lambda_\mathrm{NS}^\mathrm{Injected}=%.1f$' %\
          (Lambda),\
        levelspacing=.5, cbfmt='%.0f', cmap=cm.autumn,\
        vmin=snrThresh.min(), vmax=snrThresh.max(), xmin=-0.5, xmax=0.75, \
        figname=os.path.join(plotdir, simstring+\
      ('SNRThresholdFor%dLambdaMeasurement_BHspin_BHmass_Lambda%.1f_CI%.1f' %\
          (int(error_p), Lambda, CILevels[CI])).replace('.','_')+'.'+figtype))


if plot_LambdaCrit and recover_tidal:
  print \
"""
Assuming that larger the NS Lambda, the more easy it is to measure the NS's 
disruption's effect on the gravitational-waves emitted.

Plotting the critical value of Lambda, for different SNR levels, above which
the fractional error in LAmbda is < 100%.
"""
  #
  lambdaThresholds = {}
  #
  for snr in SNRvec:
    lambdaThresholds[snr] = {}
    for CI in range( len(CILevels) ):
      lambdaThresholds[snr][CI] = np.zeros( (len(qvec), len(chi2vec)) )
      for i, q in enumerate(qvec):
        for j, chiBH in enumerate(chi2vec):
          if verbose:
            print "getting Lambda thresholds for q=%f, chiBh=%f, SNR=%f" % \
                        (q, chiBH, snr)
          lambdaThresholds[snr][CI][i,j], fn = \
              get_Lambda_where_quantity_val_reached(data, q=q, chiBH=chiBH,\
                  SNR=snr, p='Lambda', qnt='CIfwidth', CI=CI,\
                  target_val=error_threshold, Lambdavec=Lambdavec)
          if fn > 1.e-2:
              print "\n\n warning DLambda only bound to %f %%" % (100*(fn+1))
              print "\t for q=%f, chiBh=%f, SNR=%f" % (q, chiBH, snr)
              #lambdaThresholds[snr][CI][i,j] = np.max(Lambdavec)
  #
  for snr in SNRvec:
    for CI in range( len(CILevels) ):
      if verbose:
        print "Plotting for SNR injected = %f, at Confidence level = %f" %\
                (snr, CILevels[CI])
        
      lambdaThresh = lambdaThresholds[snr][CI]
      try:
        make_contour(chi2vec, qvec, lambdaThresh,\
          xlabel='Black-hole spin', ylabel='Binary mass-ratio',\
          clabel='$\Lambda_\mathrm{NS}$ below which $\delta\Lambda_\mathrm{NS}\sim %d\%$' % int(error_p),\
          title='$\\rho^\mathrm{Injected}=%.1f$' %\
              (snr),\
          levelspacing=3, cbfmt='%.0f', cmap=cm.autumn_r,\
          #vmin=snrThresh.min(), vmax=snrThresh.max(), \
          figname=os.path.join(plotdir, simstring+\
          ('LambdaThresholdFor%dLambdaMeasurement_BHspin_MassRatio_SNR%.1f_CI%.1f' %\
              (int(error_p), snr, CILevels[CI])).replace('.','_')+'.'+figtype))
      except ValueError: 
        if verbose: print "Could not make contours for SNR=%f, CI=%d" % (snr, CI)
        pass
      #
      try:
        make_contour(chi2vec, np.array(qvec) * mNS, lambdaThresh,\
          xlabel='Black-hole spin', ylabel='Black-hole mass $(M_\odot)$',\
          clabel='$\Lambda_\mathrm{NS}$ below which $\delta\Lambda_\mathrm{NS}\sim %d\%$' % int(error_p),\
          title='$\\rho^\mathrm{Injected}=%.1f$' %\
              (snr),\
          levelspacing=3, cbfmt='%.0f', cmap=cm.autumn_r,\
          #vmin=snrThresh.min(), vmax=snrThresh.max(), \
          figname=os.path.join(plotdir, simstring+\
          ('LambdaThresholdFor%dLambdaMeasurement_BHspin_BHmass_SNR%.1f_CI%.1f' %\
              (int(error_p), snr, CILevels[CI])).replace('.','_')+'.'+figtype))
      except ValueError: 
        if verbose: print "Could not make contours for SNR=%f, CI=%d" % (snr, CI)
        pass
      


if plot_LambdaRecoveryCurves and recover_tidal:
  print """\
\n
Making summary plots, one per mass-ratio. 
Each plot shows the X% confidence interval width versus injection SNR
3 sets of curves are shown:

1. Lambda =  800, chiBH = 0, 0.5, 0.75
2. Lambda = 1000, chiBH = 0, 0.5, 0.75
3. Lambda = 1500, chiBH = 0, 0.5, 0.75

  """
  #
  print """\
First making plots showing Lambda recovery error as a function of SNR, for
different combinations of BH spins and Lambda itself.
  """
  #
  lambdaErrors = {}
  #
  for i, q in enumerate(qvec):
    lambdaErrors[q] = {}
    for CI in range( len(CILevels) ):
      lambdaErrors[q][CI] = {}
      labels = {}
      for j, chiBH in enumerate(plotchi2vec):
        lambdaErrors[q][CI][chiBH] = {}
        labels[chiBH] = {}
        for k, Lambda in enumerate(plotLambdavec):
          snr, values = get_results_vs_snr(data, q=q, chiBH=chiBH, NSLmbd=Lambda,\
                              p='Lambda', qnt='CIfwidth', CI=CI, SNRvec=SNRvec)
          lambdaErrors[q][CI][chiBH][Lambda] = [snr, values]
          labels[chiBH][Lambda] = '$\chi_\mathrm{BH}=%.2f, \Lambda=%.0f$' % (chiBH, Lambda)

      try:
        make_multilines(lambdaErrors[q][CI], labels=labels,\
          xlabel='SNR',\
          ylabel='$100\\times\delta\Lambda^{%.1f \%%}_\mathrm{NS}/\Lambda_\mathrm{NS}$' % CILevels[CI],\
          title='$q=%.1f$' % q,\
          figname=os.path.join(plotdir, simstring+\
          ('LambdaErrorVsSNRForq%dLambdaMeasurement_BHspin_BHmass_CI%.1f' %\
              (int(q), CILevels[CI])).replace('.','_')+'.'+figtype))
      except ValueError: 
        if verbose: print "Could not make line plots for Q=%f, CI=%d" % (q, CI)
        pass
  #
  print """\
Second making plots showing Lambda recovery error as a function of SNR, for
different combinations of BH spins and mass-ratio.
  """
  #
  lambdaErrors = {}
  #
  for i, Lambda in enumerate(Lambdavec):
    lambdaErrors[Lambda] = {}
    for CI in range( len(CILevels) ):
      lambdaErrors[Lambda][CI] = {}
      labels = {}
      for j, q in enumerate(plotqvec):
        lambdaErrors[Lambda][CI][q] = {}
        labels[q] = {}
        for k, chiBH in enumerate(plotchi2vec):
          snr, values = get_results_vs_snr(data, q=q, chiBH=chiBH, NSLmbd=Lambda,\
                              p='Lambda', qnt='CIfwidth', CI=CI, SNRvec=SNRvec)
          lambdaErrors[Lambda][CI][q][chiBH] = [snr, 100*values]
          labels[q][chiBH] = '$q=%.0f, \chi_\mathrm{BH}=%.2f$' % (q, chiBH)

      try:
        make_multilines(lambdaErrors[Lambda][CI], labels=labels,\
          xlabel='SNR',\
          ylabel='$100\\times\delta\Lambda^{%.1f \%%}_\mathrm{NS}/\Lambda_\mathrm{NS}$' % CILevels[CI],\
          title='$\Lambda=%.0f$' % Lambda,\
          ymax=700.,\
          single_legend=False,\
          figname=os.path.join(plotdir, simstring+\
          ('LambdaErrorVsSNRForLambda%.0fLambdaMeasurement_BHspin_BHmass_CI%.1f' %\
              (Lambda, CILevels[CI])).replace('.','_')+'.'+figtype))
      except ValueError: 
        if verbose: print "Could not make Line plots for LAMBDA=%f, CI=%d" % (Lambda, CI)
        pass


if plot_EtaCriticalCurves and recover_tidal:
  print """\

Now plotting the same as above but as a function of Lambda, and in groups of 
BH spin & SNR
  """
  EtaCritical = {}
  
  for CI in range( len(CILevels) ):
    EtaCritical[CI] = {}
    for i, Lambda in enumerate(plotLambdavec):
      EtaCritical[CI][Lambda] = {}
      labels[Lambda] = {}
      for j, snr in enumerate(plotSNRvec):
        chicritical = np.array([])
        for k, q in enumerate(plotqvec):
          # FIXME
          chicrit, fn = get_parameter_where_quantity_val_reached(
                                  data, q=q, NSLmbd=Lambda, SNR=snr,\
                                  xp='chiBH', xvec=chi2vec, \
                                  p='Lambda', qnt='CIfwidth', \
                                  CI=CI, target_val=error_threshold)
          chicritical = np.append(chicritical, chicrit)
        #
        EtaCritical[CI][Lambda][snr] = [qvec, chicritical]
        labels[Lambda][snr] = '$\Lambda=%.0f, \\rho=%.0f$' % (Lambda, snr)
    #
    try:
      make_multilines(EtaCritical[CI], labels=labels,\
          xlabel='$q=m_1/m_2$',\
          ylabel='$\chi_\mathrm{BH}^\mathrm{crit} : 100\\times\delta\Lambda^{%.1f \%%}_\mathrm{NS}/\Lambda_\mathrm{NS} < %.0f\%%$' % (CILevels[CI], error_threshold*100),\
          title='', ymin=0,\
          figname=os.path.join(plotdir, simstring+\
            ('EtaCriticalVsQForLambdaMeasurement_CI%.1f' % CILevels[CI]).replace('.','_')+'.'+figtype))
    except ValueError as val:
      print "Error :  \n", val 
      if verbose: print "Could not make Multiline plots for SNR=%f, CI=%d" % (snr, CI)
      pass











































































######################################################
# Make parameter bias plots
######################################################
#############################
##### Chirp mass bias
if plot_bias:
  print "Making chirp mass bias plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = -0.0002, 0.00035
  make_bias_plot(plotparam='Mc', ylabel='$\Delta\mathcal{M}_c / \mathcal{M}_c $',\
              savefig='plots/'+simstring+'chirpMassBias_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# Chirp mass width
if plot_width:
  print "Making chirp mass CI width plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = 0.0, 0.0012
  make_bias_plot(plotparam='Mc', plotquantity='width',\
              ylabel='$\left(\Delta\mathcal{M}_c\\right)_{90\%} / \mathcal{M}_c$',\
              savefig='plots/'+simstring+'chirpMassCIWidth90_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

#############################
##### eta bias
if plot_bias:
  print "Making eta bias plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = -0.1, 0.06
  make_bias_plot(plotparam='eta', ylabel='$\Delta\eta / \eta $',\
              savefig='plots/'+simstring+'EtaBias_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# eta width
if plot_width:
  print "Making eta CI width plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = 0.02, 0.36
  make_bias_plot(plotparam='eta', plotquantity='width',\
              ylabel='$\left(\Delta\eta\\right)_{90\%} / \eta $',\
              savefig='plots/'+simstring+'EtaCIWidth90_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

#############################
##### Chi-BH bias
if recover_tidal: plottag = 'chi1'
else: plottag = 'chi2'
if plot_bias:
  print "Making BH spin bias plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = -0.3, 0.15
  make_bias_plot(plotparam=plottag, ylabel='$\Delta \chi_\mathrm{BH}$',\
              savefig='plots/'+simstring+'BHspinBias_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# Chi-BH width
if plot_width:
  print "Making BH spin CI width plots"
  xmin, xmax = min(SNRvec), max(SNRvec)
  ymin, ymax = 0.0, 0.9
  make_bias_plot(plotparam=plottag, plotquantity='width',\
              ylabel='$\left(\Delta \chi_\mathrm{BH}\\right)_{90\%}$',\
              savefig='plots/'+simstring+'BHspinCIWidth90_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


#############################
# Lambda-NS bias & width
if recover_tidal:
  plottag = 'chi2'
  if plot_bias:
    print "Making NS Lambda bias plots"
    xmin, xmax = min(SNRvec), max(SNRvec)
    ymin, ymax = -0.2, 2.5
    make_bias_plot(plotparam=plottag,\
              normalize='Lambda',\
              ylabel='$\Delta\Lambda_\mathrm{NS} / \Lambda_\mathrm{NS}$',\
              savefig='plots/'+simstring+'NSLambdaBias_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
  
  if plot_width:
    print "Making NS Lambda CI Width plots"
    xmin, xmax = min(SNRvec), max(SNRvec)
    ymin, ymax = 0, 7
    make_bias_plot(plotparam=plottag, plotquantity='width',\
        normalize='Lambda',\
        ylabel='$\left(\Delta\Lambda_\mathrm{NS}\\right)_{90\%} / \Lambda_\mathrm{NS}$',\
              savefig='plots/'+simstring+'NSLambdaCIWidth90_vs_SNR_q23.pdf',\
              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)






