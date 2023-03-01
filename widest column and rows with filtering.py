# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 10:06:44 2023

@author: limyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths
import statistics

def calculate_statistics(array):
    mean = np.mean(array)
    std_dev = np.std(array)
    mse = np.mean((array - mean)**2)
    return mean, std_dev, mse

def second_highest(arr):
  sorted_arr = sorted(set(arr), reverse=True)
  return sorted_arr[1] if len(sorted_arr) > 1 else None

df = pd.read_csv(r"C:\Users\limyu\Google Drive\focusing grating\2D grating GDS\grating type 4 branchs\different radius lower 15um 20um 30um aligning the ref 4x 40 mA_0001.ascii.csv",header=None, sep=",")
     
df=df.dropna(axis=1)
df_r = df.iloc[220:244, 220:254]
df_r = df_r.reset_index(drop=True)
H = np.arange(0,24,1)
V = np.arange(0,34,1)
n = 4 #how many times highest peak should be taller than second highest peak
fraction = 0.8

verticle_data = pd.DataFrame([])
for v in V:
    x = df_r.iloc[:,v]
    x = x - min(x)
    x = x.to_numpy()
    peaks, _ = find_peaks(x)
    results_full = peak_widths(x, peaks, rel_height=0.865)
    results_full[0]  # widths
    
    if (len(x[peaks]) <=1 or n*(second_highest(x[peaks])) < max(x[peaks])) and max(x) > fraction*df_r.max().max():
        verticle_data[str(v)] = x


widest_peak_each_column_list = []
index_v = []

for col in verticle_data.columns:
    x1 = verticle_data[col].values
    peaks_v_narrow, _ = find_peaks(x1)
    results_full_narrow = peak_widths(x1, peaks_v_narrow, rel_height=0.865)
    widest_peak_each_coloumn = max(results_full_narrow[0])
    widest_peak_each_column_list.append(widest_peak_each_coloumn)
    index_v.append(col)
widest_column = widest_peak_each_column_list.index(max(widest_peak_each_column_list))
widest_verticle = index_v[widest_column]



horizontal_data = pd.DataFrame([])
for h in H:
    x = df_r.iloc[h,:]
    x = x - min(x)
    x = x.to_numpy()
    peaks, _ = find_peaks(x)

    results_full = peak_widths(x, peaks, rel_height=0.865)
    results_full[0]  # widths
    
    if (len(x[peaks]) <=1 or n*(second_highest(x[peaks])) < max(x[peaks])) and max(x) > fraction*df_r.max().max():
        horizontal_data[str(h)] = x


widest_peak_each_column_list = []
index_h = []

for col in horizontal_data.columns:
    x1 = horizontal_data[col].values
    peaks_v_narrow, _ = find_peaks(x1)
    results_full_narrow = peak_widths(x1, peaks_v_narrow, rel_height=0.865)
    widest_peak_each_coloumn = max(results_full_narrow[0])
    widest_peak_each_column_list.append(widest_peak_each_coloumn)
    index_h.append(col)
widest_column = widest_peak_each_column_list.index(max(widest_peak_each_column_list))
widest_horizontal = index_h[widest_column]

print(" ")
print("the widest vertical line is:", widest_verticle)
print("the widest horizontal line is:", widest_horizontal)
print(" ")

xr = np.linspace(0, 990, num=34)
xr = xr/20
yr = np.linspace(0, 690, num=24)
yr = yr/20
colorbarmax = 5000
colorbartick = 5

Xr,Yr = np.meshgrid(xr,yr)
df_r = df_r.to_numpy()
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
cp=ax.contourf(Xr,Yr,df_r, 200, zdir='z', offset=-100, cmap='hot')
clb=fig.colorbar(cp, ticks=(np.arange(0, colorbarmax, 500)).tolist())
clb.ax.set_title('Photon/s', fontweight="bold")
for l in clb.ax.yaxis.get_ticklabels():
    l.set_weight("bold")
    l.set_fontsize(15)
ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
ax.set_ylabel('y-position (µm)', fontsize=18, fontweight="bold", labelpad=1)
ax.xaxis.label.set_fontsize(18)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(18)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.set_xticklabels(ax.get_xticks(), weight='bold')
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax.axhline(y=yr[int(widest_horizontal)], color='r', linestyle = "--")
ax.axvline(x=xr[int(widest_verticle)], color='g', linestyle = "--")
plt.show()
plt.close()

V1,H1 = np.meshgrid(V,H)
fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
cp=ax.contourf(V1,H1,df_r, 200, zdir='z', offset=-100, cmap='hot')
ax.axhline(y=int(widest_horizontal), color='r')
ax.axvline(x=int(widest_verticle), color='g')
plt.show()
plt.close()