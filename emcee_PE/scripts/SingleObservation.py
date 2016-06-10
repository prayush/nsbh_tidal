#!/usr/bin/env python
# Tools for analysis of MCMC samples from emcee for spin estimation project
# Copyright 2015
# 
__author__ = "Prayush Kumar <prayush.kumar@ligo.org>"

import os, sys
import commands as cmd
import numpy as np
from matplotlib import mlab, cm, use
use('pdf')
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
ErrThresh=[100, 200]

######################################################
# PLOTTING :
# Function to make parameter bias plots
######################################################
linestyles = ['dotted', 'dashed', 'solid', 'dashed', 'dotted', 'dashdot', 'dashed', 'dotted']
#linecolors = ['crimson', 'olivedrab', 'k', 'b', 'm', 'y']
linecolors = ['crimson', 'darkorange', 'olivedrab', 'royalblue', 'purple', 'k']
linemarkers= ['', 'x', 'o', '^']

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
        title='', titles=[], xticks=None, yticks=None,\
	levellinesscale=None, levellines=[100,200], levellabels=True, levelspacing=0.25,\
        vmin=None, vmax=None, cmap=cm.rainbow, cfmt='%.1f', lfmt='%.1f',\
        xmin=None, xmax=None, ymin=None, ymax=None, alpha=0.9,\
        colorticks=None, colorbartype='simple', figname='plot.png'):
  """
  Function to plot arbitrary numbers of contour plots in a single figure
  """
  print """\

Remember that the returned set of contours are indexed by integer pairs that
set coordinates for [row, col]:-

row corresponds to the higher level group in the input Z array, 
col corresponds to the lower level group

  """
  if colorbartype != 'simple':
    raise IOError("Colorbar type %s not supported" % colorbartype)
  #
  if np.shape(X)[:2] != np.shape(Y)[:2] or np.shape(X)[:2] != np.shape(Z2d)[:2]:
    raise IOError("X, Y and Z arrays have different number of sets to plot")
  
  plt.rcParams.update({\
    'legend.fontsize':16, \
    'text.fontsize':16,\
    'axes.labelsize':16,\
    'font.family':'serif',\
    'font.size':16,\
    'xtick.labelsize':14,\
    'ytick.labelsize':14,\
    'figure.subplot.bottom':0.2,\
    'figure.figsize':figsize, \
    'savefig.dpi': 300.0})#, \
  #'figure.autolayout': True})

  nrow, ncol, _ = np.shape(X)
  if vverbose: print "Making plot with %d rows, %d cols" % (nrow, ncol)
  pltid = 0
  
  colwidth = 1.3
  fig = plt.figure(int(1e7 * np.random.random()), \
              figsize=((1.3*gmean*ncol+1.75)*colwidth, 1.2*colwidth*nrow),\
              dpi=100)
  fig.clf()
  grid = ImageGrid(fig, 111,\
            nrows_ncols=(nrow, ncol), \
            share_all=True,\
            axes_pad=0.2,\
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
  if vverbose: print "VMIN = %e, VMAX = %e" % (VMIN, VMAX)
  if VMIN > VMAX: raise IOError("Cant fit all data on a single colorbar")
  
  ## FIXME
  #VMIN = 0.01
  
  # Now make contour plots
  contours_all = {}
  for idx in range(nrow):
    contours_all[idx] = {}
    for jdx in range(ncol):
      levline_scaling = 1
      try:
        if levellinesscale is not None:
          levline_scaling = levellinesscale[idx][jdx]
      except: pass
      #
      print "Scaling level lines with %f" % levline_scaling
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
              alpha=alpha, vmin=VMIN, vmax=VMAX)
      contours_tmp = {}
      for lev in levellines:
        cset = ax.contour(xx, yy, zz, levels=[lev * levline_scaling],\
			colors='k', ls='--', linewidths=1.2, hold="on")
        for c in cset.collections: c.set_linestyle('dotted')
        if levellabels:
          label_dict = {}
          for ll in [lev * levline_scaling]: label_dict[ll] = str(int(lev*100))
          lfmt = label_dict
        plt.clabel(cset, colors='r', inline=1, fmt=lfmt, fontsize=10)
        contours_tmp[lev] = cset.collections[0].get_paths()
      if vverbose:
        print "for %s" % titles[idx][jdx], contours_tmp
        print "ymin, ymax = ", ymin, ymax
      #
      ax.grid(True, color='k')
      if xticks is not None: ax.get_xaxis().set_ticks(xticks)
      if yticks is not None: ax.get_yaxis().set_ticks(yticks)
      ax.set_xlim([xmin, xmax])
      ax.set_ylim([ymin, ymax])
      if idx == (nrow-1): ax.set_xlabel(xlabel)
      if jdx == 0: ax.set_ylabel(ylabel)
      #if np.shape(titles) == (nrow, ncol): ax.set_title(titles[idx][jdx])
      if np.shape(titles) == (nrow, ncol):
        ax.text(.5, .8, titles[idx][jdx], horizontalalignment='center', transform=ax.transAxes)
      #if idx == 0 and jdx==(ncol/2) and title != '':
      #  ax.set_title(titles[idx][jdx]+'\n '+title)  
      if idx == 0 and jdx==(ncol/2) and title != '':
        ax.text(.5, .8, titles[idx][jdx]+'\n '+title,\
          horizontalalignment='center', transform=ax.transAxes)
      #
      contours_all[idx][jdx] = contours_tmp
    if vverbose:
      print "Made ROW %d/%d" % (idx+1, nrow)
  #
  if colorticks is not None: cb = ax.cax.colorbar(CS, format=cfmt, ticks=colorticks)
  else: cb = ax.cax.colorbar(CS, format=cfmt)
  cb.set_label_text(clabel)
  ax.cax.toggle_label(True)
  fig.subplots_adjust(right=0.8)
  #fig.tight_layout(rect=(0, 0, 0.82, 1))
  fig.savefig(figname, dpi=300)
  return contours_all


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
              alpha=alpha, vmin=vmin, vmax=vmax)
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
  colwidth = 4.
  plt.figure(int(1e7 * np.random.random()), figsize=(0.75*gmean*colwidth, 0.75*colwidth))
  norm = cm.colors.Normalize(vmax=Z2d.max(), vmin=Z2d.min())
  #cmap = cm.rainbow
  levels = np.arange(Z2d.min(), Z2d.max(), levelspacing)
  print np.min(Z2d), np.max(Z2d), levelspacing
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
                levelspacing=0.25, vmin=None, vmax=None, background_color=None,\
                pcolors=linecolors, pmarkers=linestyles, lw=1.5,\
                xmin=None, xmax=None, ymin=None, ymax=None,\
                markerfirst=[True, False], sort_keys=False,\
                single_legend=True, leg_loc=[1,2],\
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
    'savefig.dpi': 500.0, \
    'figure.autolayout': True})
  
  ngrp1 = len(XY.keys())
  ngrp2 = len(XY[XY.keys()[0]].keys())
  if ngrp1 > len(pcolors) or ngrp2 > len(pmarkers):
    raise IOError("More lines to be made than colors/markers given")
  if vverbose: print "Making plot with %d groups of %d lines" % (ngrp1, ngrp2)
    
  #fig = plt.figure(int(1e7 * np.random.random()), \
  #            figsize=((2.1*gmean*ncol+1.25)*colwidth, 1.2*colwidth*nrow))
  fig = plt.figure(int(1e7 * np.random.random()), figsize=(1.2*gmean*colwidth*0.75, 0.75*colwidth))
  # FIrst make all lines
  all_lines = []
  xykeys = XY.keys()
  if sort_keys: xykeys.sort()
  for i, ki in enumerate(xykeys):
    grp_lines = []
    xykkeys = XY[ki].keys()
    if sort_keys: xykkeys.sort()
    for j, kj in enumerate(xykkeys):
      X, Y = XY[ki][kj]
      line, = plt.plot(X, Y, c=pcolors[i], ls=pmarkers[j], lw=lw, markersize=7, label=labels[ki][kj])
      grp_lines.append( line )
    all_lines.append( grp_lines )
  #
  if background_color is not None: plt.gca().get_axes().set_axis_bgcolor(background_color)
  # Now make legends, one to indicate each group's characteristic
  if single_legend:
    harray = [all_lines[i][0] for i in range(ngrp1)]
    harray.extend( [all_lines[0][i] for i in range(ngrp2)] )
    first_legend = plt.legend(handles=harray, loc=1, ncol=2, framealpha=False, markerfirst=markerfirst[0], fontsize=10)
    ax = plt.gca().add_artist(first_legend)
  else:
    harray1 = [all_lines[i][0] for i in range(ngrp1)]
    harray2 = [all_lines[0][i] for i in range(ngrp2)]
    first_legend  = plt.legend(handles=harray1, loc=leg_loc[0], ncol=1, framealpha=False, markerfirst=markerfirst[0], fontsize=10)
    ax = plt.gca().add_artist(first_legend)
    second_legend = plt.legend(handles=harray2, loc=leg_loc[1], ncol=1, framealpha=False, markerfirst=markerfirst[1], fontsize=10)
  #
  plt.grid(alpha=0.5)
  plt.xlim([xmin,xmax])
  plt.ylim([ymin,ymax])
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.savefig(figname)
  return

def make_multilines_3(XY, labels=None, xlabel='SNR', ylabel='', clabel='', title='',\
                levelspacing=0.25, vmin=None, vmax=None, cmap=cm.Reds_r, logY=True,\
                pcolors=linecolors, plines=linestyles, pmarkers=linemarkers,\
                alpha=0.8,\
                xmin=None, xmax=None, ymin=None, ymax=None,\
                constant_line=None,\
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
  ngrp3 = len(XY[XY.keys()[0]][XY[XY.keys()[0]].keys()[0]].keys())
  if ngrp1 > len(pcolors) or ngrp2 > len(pmarkers) or ngrp3 > len(plines):
    raise IOError("More lines to be made than colors/markers given")
  if vverbose:
    print "Making plot with %d groups of %d groups of %d lines" %\
            (ngrp1, ngrp2, ngrps)
    
  #fig = plt.figure(int(1e7 * np.random.random()), \
  #            figsize=((2.1*gmean*ncol+1.25)*colwidth, 1.2*colwidth*nrow))
  plt.figure(int(1e7 * np.random.random()), figsize=(1.*gmean*colwidth, colwidth))
  
  # FIrst make all lines
  all_lines = []
  for i, ki in enumerate(XY.keys()):
    grp1_lines = []
    for j, kj in enumerate(XY[ki].keys()):
      grp2_lines = []
      for k, kk in enumerate(XY[ki][kj].keys()):
        X, Y = XY[ki][kj][kk]
        if logY:
          line, = plt.semilogy(X, Y,\
                          alpha=alpha,\
                          c=pcolors[i], ls=plines[j], marker=pmarkers[k],\
                          lw=.5, markersize=3, label=labels[ki][kj][kk])
        else:
          line, = plt.plot(X, Y,\
                          alpha=alpha,\
                          c=pcolors[i], ls=plines[j], marker=pmarkers[k],\
                          lw=.5, markersize=3, label=labels[ki][kj][kk])
        grp2_lines.append( line )
      grp1_lines.append( grp2_lines )
    all_lines.append( grp1_lines )
  if constant_line is not None and logY:
    print "Adding constant line with ", [xmin, xmax], constant_line
    plt.semilogy([xmin, xmax], constant_line, 'k', lw=2)
  elif constant_line is not None:
    plt.plot([xmin, xmax], constant_line, 'k', lw=2)
  #
  # Now make legends, one to indicate each group's characteristic
  if single_legend:
    harray = [all_lines[i][0][0] for i in range(ngrp1)]
    harray.extend( [all_lines[0][i][0] for i in range(ngrp2)] )
    harray.extend( [all_lines[0][0][i] for i in range(ngrp3)] )
    first_legend = plt.legend(handles=harray, loc=1, ncol=2, framealpha=False)
    ax = plt.gca().add_artist(first_legend)
  else:
    harray1 = [all_lines[i][0][0] for i in range(ngrp1)]
    harray2 = [all_lines[0][i][0] for i in range(ngrp2)]
    harray3 = [all_lines[0][0][i] for i in range(ngrp3)]
    first_legend = plt.legend(handles=harray1, loc=4, ncol=1, framealpha=False)
    ax = plt.gca().add_artist(first_legend)
    second_legend = plt.legend(handles=harray2, loc=2, ncol=1, framealpha=False, markerfirst=True)
    ax2 = plt.gca().add_artist(second_legend)
    third_legend = plt.legend(handles=harray3, loc=3, ncol=1, framealpha=False, markerfirst=False)
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
# DATA PROCESSING :
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
        CI=0, CILevs=[90.0, 68.26895, 95.44997, 99.73002], fmt='new'):
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
  if 'old' in fmt:
    l1grp, l2grp, l3grp = \
        ['q%.1f.dir' % q, 'chiBH%.2f.dir' % chiBH,\
         'LambdaNS%.1f.dir' % NSLmbd]
    dset = 'SNR%.1f.dat' % SNR
    alldata = data[l1grp][l2grp][l3grp][dset].value
    if vverbose: print "Group names = ", l1grp, l2grp, l3grp, l4grp, dset
  else:
    l1grp, l2grp, l3grp, l4grp = \
        ['q%.1f.dir' % q, 'chiBH%.2f.dir' % chiBH,\
         'LambdaNS%.1f.dir' % NSLmbd, 'SNR%.1f.dir' % SNR]
    dset = 'summary.dat'
    alldata = data[l1grp][l2grp][l3grp][l4grp][dset].value
    if vverbose: print "Group names = ", l1grp, l2grp, l3grp, l4grp, dset
  
  
  if 'old' in fmt: num_of_data_fields = 5
  else: num_of_data_fields = 7
  
  if 'm1' in p: pidx = 2 + 0*num_of_data_fields
  elif 'm2' in p: pidx = 2 + 1*num_of_data_fields
  elif 'Mc' in p: pidx = 2 + 2*num_of_data_fields
  elif 'Mtot' in p: pidx = 2 + 3*num_of_data_fields
  elif 'eta' in p: pidx = 2 + 4*num_of_data_fields
  elif 'q' in p: pidx = 2 + 5*num_of_data_fields
  elif 'chiBH' in p: pidx = 2 + 6*num_of_data_fields
  elif 'Lambda' in p: pidx = 2 + 7*num_of_data_fields
  
  if 'old' in fmt:
    if 'median-val' in qnt: pidx += 0
    elif 'fbias' == qnt: pidx += 1
    elif 'CIlower' in qnt: pidx += 2
    elif 'CIhigher' in qnt: pidx += 3
    elif 'CIfwidth' in qnt: pidx += 4
  else:
    if 'median-val' in qnt: pidx += 0
    elif 'maxLogL-val' in qnt: pidx += 1
    elif 'fbias' in qnt: pidx += 2
    elif 'maxLogL-fbias' in qnt: pidx += 3
    elif 'CIlower' in qnt: pidx += 4
    elif 'CIhigher' in qnt: pidx += 5
    elif 'CIfwidth' in qnt: pidx += 6
  
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


