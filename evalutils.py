#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:12:29 2022

@author: sadams
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_bland_altman(trueMeas, compMeas, trueLabel, compLabel, heading, rng=None, regressionPlot=True, units=None):
    if units is None:
        unitStr = ""
    else:
        unitStr = " (" + units + ")"
    N = len(trueMeas)
    
    if rng is None:
        minMeas = np.min([np.min(trueMeas), np.min(compMeas)])
        maxMeas = np.max([np.max(trueMeas), np.max(compMeas)])
    else:
        minMeas, maxMeas = rng[0], rng[1]
    
    xlim = np.array([minMeas, maxMeas])
    
    if regressionPlot:
        plt.figure(figsize=(8, 6), dpi=150)
        plt.plot(trueMeas, compMeas, 'x')
        
        slope, intercept, r_value, p_value, std_err = linregress(trueMeas, compMeas)
        ylin = slope*xlim + intercept
        
        plt.plot(xlim, xlim, 'k--')
        plt.plot(xlim, ylin, 'k-')
        plt.text(minMeas, maxMeas,
                 f'\n N={N} \n y={slope:.3f}x + {intercept:.3f} \n R2={r_value**2:.3f}',
                 horizontalalignment='left', verticalalignment='top')
        
        plt.xlim((minMeas, maxMeas)); plt.ylim((minMeas, maxMeas))
        plt.xlabel(trueLabel + unitStr); plt.ylabel(compLabel + unitStr)
        plt.title(heading + " Regression Analysis")
        plt.show()
    
    diff = compMeas - trueMeas
    diffMean, diffStd = np.mean(diff), np.std(diff)
    diffBound = np.max(np.abs(diff))
    
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(trueMeas, diff, 'x')
    
    diffUp, diffDown = diffMean + 1.96*diffStd, diffMean - 1.96*diffStd
    plt.plot(xlim, [diffMean, diffMean], 'k-')
    plt.plot(xlim, [diffUp, diffUp], 'k--')
    plt.plot(xlim, [diffDown, diffDown], 'k--')
    plt.text(minMeas, diffBound,
             f'\n mean={diffMean:.1f} \n st.dev.={diffStd:.1f} \n 95%CI=({diffDown:.1f}, {diffUp:.1f})',
             horizontalalignment='left', verticalalignment='top')
    
    plt.xlim((minMeas, maxMeas)); plt.ylim((-diffBound, diffBound))
    plt.xlabel(trueLabel + unitStr); plt.ylabel(compLabel + " - " + trueLabel + unitStr)
    plt.title(heading + " Bland-Altman Analysis")
    plt.show()
    
def slice_results(sli: int, t1map, t2map, t1pred, t2pred, t1rng=(0, 2000), t2rng=(0, 250)):
    # Plot slices
    plt.figure(figsize=(8, 6), dpi=150)
    
    if t1map.shape[2] < 2:
        sli = None

    plt.subplot(2, 2, 1)
    plt.imshow(t1pred[:, :, sli], vmin=t1rng[0], vmax=t1rng[1])
    plt.colorbar()
    plt.xticks([]); plt.yticks([])
    plt.title("Neural T1 map")
    
    plt.subplot(2, 2, 2)
    plt.imshow(t1map[:, :, sli], vmin=t1rng[0], vmax=t1rng[1])
    plt.colorbar()
    plt.xticks([]); plt.yticks([])
    plt.title("STIR T1 map")

    plt.subplot(2, 2, 3)
    plt.imshow(t2pred[:, :, sli], vmin=t2rng[0], vmax=t2rng[1])
    plt.colorbar()
    plt.xticks([]); plt.yticks([])
    plt.title("Neural T2 map")

    plt.subplot(2, 2, 4)
    plt.imshow(t2map[:, :, sli], vmin=t2rng[0], vmax=t2rng[1])
    plt.colorbar()
    plt.xticks([]); plt.yticks([])
    plt.title("SE T2 map")

    plt.suptitle(f"Slice {sli + 1}")

    plt.show()