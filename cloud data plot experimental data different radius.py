# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 17:21:33 2023

@author: limyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths

def gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y

master_data_verticle = pd.DataFrame([])
master_data_horizontal = pd.DataFrame([])
full_width_horizontal = []
full_width_verticle = []
half_width_horizontal = []
half_width_verticle = []
verticle_peaks = []
horizontal_peaks = []
horizontal_peaks_position = []
verticle_peaks_position = []
horizontal_width_cut = []
verticle_width_cut = []
cut = 1200

df = pd.read_csv("https://raw.githubusercontent.com/yd145763/different_radius_publication_data_code/main/2nd%20march%20different%20radius%20lower%2015%2020%2030%20um%20aligning%20the%20ref%203800-3900cnts%204x%2055mA%20mA_0001.ascii.csv")


verticle_peaks = []
verticle_half = []
verticle_full = []

horizontal_peaks = []
horizontal_half = []
horizontal_full = []
def plot_large(df):
    x = np.linspace(0, 9570, num=df.shape[1])
    x = x/20
    y = np.linspace(0, 7650, num=df.shape[0])
    y = y/20
    colorbarmax = df.max().max()

    
    X,Y = np.meshgrid(x,y)
    df1 = df.to_numpy()
    
    fig = plt.figure(figsize=(7, 4))
    ax = plt.axes()
    cp=ax.contourf(X,Y,df, 200, zdir='z', offset=-100, cmap='hot')
    clb=fig.colorbar(cp, ticks=(np.arange(0, colorbarmax, round(colorbarmax/5, -2))).tolist())
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
    plt.show()
    plt.close()
    
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')
    ax.plot_surface(X, Y, df1, cmap='hot')
    ax.set_xlabel('x-position (µm)', fontsize=18, fontweight="bold", labelpad=13)
    ax.set_ylabel('y-position (µm)', fontsize=18, fontweight="bold", labelpad=15)
    ax.set_zlabel('Photon/s', fontsize=18, fontweight="bold", labelpad=15)
    
    
    ax.xaxis.label.set_fontsize(18)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(18)
    ax.yaxis.label.set_weight("bold")
    ax.zaxis.label.set_fontsize(18)
    ax.zaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_zticklabels(ax.get_zticks(), weight='bold')
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.show()
    plt.close()
    
plot_large(df)

def slicing(r_start, r_end, col_start, col_end):
    df_sliced = df.iloc[r_start:r_end,col_start: col_end]
    return df_sliced

def operate_everything(df_r, radius):
    max_df_r = df_r.max().max()
    row_idxs, col_idxs = np.where(df_r == max_df_r)
    row_idxs = int(row_idxs)
    col_idxs = int(col_idxs)
    df_r_hor = df_r.iloc[row_idxs,:]
    df_r_ver = df_r.iloc[:, col_idxs]

    xr = np.linspace(0, 990, num=34)
    xr = xr/20
    yr = np.linspace(0, 690, num=24)
    yr = yr/20
    colorbarmax = df_r.max().max()
    Xr,Yr = np.meshgrid(xr,yr)
    df_r = df_r.to_numpy()
    fig = plt.figure(figsize=(8, 4))
    ax = plt.axes()
    cp=ax.contourf(Xr,Yr,df_r, 200, zdir='z', offset=-100, cmap='hot')
    clb=fig.colorbar(cp, ticks=(np.arange(0, colorbarmax, round(colorbarmax/5, -2))).tolist())
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
    ax.axhline(y=yr[row_idxs], color='r', linestyle = "--")
    ax.axvline(x=xr[col_idxs], color='g', linestyle = "--")
    plt.show()
    plt.close()
    
    
    #plot horizontal
    ax2 = plt.axes()
    tck = interpolate.splrep(xr, df_r_hor.tolist(), s=2, k=4) 
    x_new = np.linspace(min(xr), max(xr), 1000)
    y_fit = interpolate.BSpline(*tck)(x_new)
    h_r = y_fit 
    peaks, _ = find_peaks(h_r)
    peaks_h = x_new[peaks]
    horizontal_peaks.append(peaks_h)
    
    results_half = peak_widths(h_r, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(x_new[1] - x_new[0]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
    x_min = np.array(x_min)
    x_max = results_half[3]
    x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
    x_max = np.array(x_max)
    results_half_plot = (width, height, x_min, x_max)
    
    results_full = peak_widths(h_r, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
    x_min_f = np.array(x_min_f)
    x_max_f = results_full[3]
    x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
    x_max_f = np.array(x_max_f)
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    
    
    
    horizontal_half.append(max(results_half_plot[0]))
    horizontal_full.append(max(results_full_plot[0]))
    
    
    ax2.plot(x_new, h_r)
    ax2.plot(peaks_h, h_r[peaks], "o")
    ax2.hlines(*results_half_plot[1:], color="C2")
    ax2.hlines(*results_full_plot[1:], color="C3")
    
    
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    
    ax2.xaxis.label.set_fontsize(18)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(18)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("x-position (µm)")
    plt.ylabel("Photon/s")
    plt.legend(["Photon/s", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
    plt.show()
    plt.close()
    
    full_width_horizontal.append(max(results_full_plot[0]))
    half_width_horizontal.append(max(results_half_plot[0]))
    horizontal_peaks.append(max(h_r))
    horizontal_peaks_position.append(x_new[np.argmax(h_r)])
    master_data_horizontal[radius] = h_r
    
    #plot verticle
    ax2 = plt.axes()
    tck = interpolate.splrep(yr, df_r_ver.tolist(), s=2, k=4) 
    x_new = np.linspace(min(yr), max(yr), 1000)
    y_fit = interpolate.BSpline(*tck)(x_new)
    v_r = y_fit 
    peaks, _ = find_peaks(v_r)
    peaks_v = x_new[peaks]
    verticle_peaks.append(peaks_v)
    
    results_half = peak_widths(v_r, peaks, rel_height=0.5)
    width = results_half[0]
    width = [i*(x_new[1] - x_new[0]) for i in width]
    width = np.array(width)
    height = results_half[1]
    x_min = results_half[2]
    x_min = [i*(x_new[1] - x_new[0]) for i in x_min]
    x_min = np.array(x_min)
    x_max = results_half[3]
    x_max = [i*(x_new[1] - x_new[0]) for i in x_max]
    x_max = np.array(x_max)
    results_half_plot = (width, height, x_min, x_max)
    
    results_full = peak_widths(v_r, peaks, rel_height=0.865)
    width_f = results_full[0]
    width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
    width_f = np.array(width_f)
    height_f = results_full[1]
    x_min_f = results_full[2]
    x_min_f = [i*(x_new[1] - x_new[0]) for i in x_min_f]
    x_min_f = np.array(x_min_f)
    x_max_f = results_full[3]
    x_max_f = [i*(x_new[1] - x_new[0]) for i in x_max_f]
    x_max_f = np.array(x_max_f)
    results_full_plot = (width_f, height_f, x_min_f, x_max_f)
    
    
    verticle_half.append(max(results_half_plot[0]))
    verticle_full.append(max(results_full_plot[0]))
    
    ax2.plot(x_new, v_r)
    ax2.plot(peaks_v, v_r[peaks], "o")
    ax2.hlines(*results_half_plot[1:], color="C2")
    ax2.hlines(*results_full_plot[1:], color="C3")
    
    
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    
    ax2.xaxis.label.set_fontsize(18)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(18)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("y-position (µm)")
    plt.ylabel("Photon/s")
    plt.legend(["Photon/s", "Peaks", "FWHM", "Beam Waist"], prop={'weight': 'bold'})
    plt.show()
    plt.close()
    
    full_width_verticle.append(max(results_full_plot[0]))
    half_width_verticle.append(max(results_half_plot[0]))
    verticle_peaks.append(max(v_r))
    verticle_peaks_position.append(x_new[np.argmax(v_r)])
    master_data_verticle[radius] = v_r

r_start_r1, r_end_r1, col_start_r1, col_end_r1 = 221,245,207,241
df_r1 = slicing(r_start_r1, r_end_r1, col_start_r1, col_end_r1)
operate_everything(df_r1, "15um")

r_start_r2, r_end_r2, col_start_r2, col_end_r2 = 83,107,204,238
df_r2 = slicing(r_start_r2, r_end_r2, col_start_r2, col_end_r2)
operate_everything(df_r2, "20um")

r_start_r3, r_end_r3, col_start_r3, col_end_r3 = 13,37,204,238
df_r3 = slicing(r_start_r3, r_end_r3, col_start_r3, col_end_r3)
operate_everything(df_r3, "30um")

df = pd.read_csv("https://raw.githubusercontent.com/yd145763/different_radius_publication_data_code/main/2nd%20march%20different%20radius%20upper%2040%2050%2060%20um%20aligning%20the%20ref%203800-3900cnts%204x%2055mA%20mA_0001.ascii.csv")

plot_large(df)

r_start_r4, r_end_r4, col_start_r4, col_end_r4 = 216,240,207,241
df_r4 = slicing(r_start_r4, r_end_r4, col_start_r4, col_end_r4)
operate_everything(df_r4, "60um")

r_start_r5, r_end_r5, col_start_r5, col_end_r5 = 78,102,204,238
df_r5 = slicing(r_start_r5, r_end_r5, col_start_r5, col_end_r5)
operate_everything(df_r5, "50um")

r_start_r6, r_end_r6, col_start_r6, col_end_r6 = 10,34,204,238
df_r6 = slicing(r_start_r6, r_end_r6, col_start_r6, col_end_r6)
operate_everything(df_r6, "40um")

xr = np.linspace(0, 990, num=34)
xr = xr/20
hor = np.linspace(min(xr), max(xr), 1000)
yr = np.linspace(0, 690, num=24)
yr = yr/20
ver = np.linspace(min(yr), max(yr), 1000)

ax2 = plt.axes()
ax2.plot(hor, master_data_horizontal["15um"])
ax2.plot(hor, master_data_horizontal["20um"])
ax2.plot(hor, master_data_horizontal["30um"])
ax2.plot(hor, master_data_horizontal["40um"])
ax2.plot(hor, master_data_horizontal["50um"])
ax2.plot(hor, master_data_horizontal["60um"])
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold'})
plt.show()
plt.close()

ax2 = plt.axes()
ax2.plot(ver, master_data_verticle["15um"])
ax2.plot(ver, master_data_verticle["20um"])
ax2.plot(ver, master_data_verticle["30um"])
ax2.plot(ver, master_data_verticle["40um"])
ax2.plot(ver, master_data_verticle["50um"])
ax2.plot(ver, master_data_verticle["60um"])
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(18)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(18)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Photon/s")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold'})
plt.show()
plt.close()

print("r1 is 30µm")
print("r2 is 20µm")
print("r3 is 15µm")
print("r4 is 60µm")
print("r5 is 50µm")
print("r6 is 40µm")

exp_filtered_by_width = [20.511997, 12.2996, 13.958, 15.6158, 17.88, 18.5346 ]
exp_no_filter_max_index = [20.511997, 12.2996, 13.958, 14.207003, 17.44541, 18.5346]
simulated_40um = [15.20, 6.89, 16.33, 21.72, 24.16, 26.37]
max_photon_count = [1335, 1852, 2276, 3955, 4006, 2713]
radius = ["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"]
fig, ax2 = plt.subplots()
ax2.scatter(radius, exp_filtered_by_width, marker = "o", s=50)
ax2.scatter(radius, exp_no_filter_max_index, marker = "s", s=50)
ax2.scatter(radius, simulated_40um, marker = "v", s=50)
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(16)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(16)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
ax2.spines["right"].set_linewidth(2)
ax2.spines["top"].set_linewidth(2)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
ax2.set_ylabel("Beam Width (µm)")
ax2.set_xlabel("Radius of Curvature (µm)")
ax2.legend(["Selection by Largest Beam Width", "Selection by Highest Photon/s", "Simulated Beam Width"], loc = 'lower right', prop={'weight': 'bold'})

ax1 = ax2.twinx()
ax1.plot(radius, max_photon_count, color='blue')
ax1.set_ylabel('Peak Photon/s', color = "blue")
ax1.tick_params(which='major', width=2.00)
ax1.tick_params(which='minor', width=2.00)
ax1.xaxis.label.set_fontsize(18)
ax1.xaxis.label.set_weight("bold")
ax1.yaxis.label.set_fontsize(18)
ax1.yaxis.label.set_weight("bold")
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_yticklabels(ax1.get_yticks(), weight='bold', color = "blue")
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax1.set_xlabel("Radius of Curvature (µm)")
plt.show()
plt.close()

exp_filtered_by_width = [13.958, 12.2996, 15.6158, 17.88, 18.5346, 20.511997]
exp_no_filter_max_index = [13.958, 12.2996, 14.207003, 17.44541, 18.5346, 20.511997]
simulated_40um = [15.20, 6.89, 16.33, 21.72, 24.16, 26.37]
max_photon_count = [1335, 1852, 2276, 3955, 4006, 2713]
radius = ["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"]
fig, ax2 = plt.subplots()
ax2.scatter(radius, exp_filtered_by_width, marker = "o", s=50)
ax2.scatter(radius, exp_no_filter_max_index, marker = "s", s=50)
ax2.scatter(radius, simulated_40um, marker = "v", s=50)
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(16)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(16)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
ax2.spines["right"].set_linewidth(2)
ax2.spines["top"].set_linewidth(2)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
ax2.set_ylabel("Beam Waist (µm)")
ax2.set_xlabel("Radius of Curvature (µm)")
ax2.legend(["Column Selection by Beam Waist", "Column Selection by Intensity", "Simulated Beam Waist"], prop={'weight': 'bold'})

ax1 = ax2.twinx()
ax1.plot(radius, max_photon_count, color='blue')
ax1.set_ylabel('Peak Photon/s', color = "blue")
ax1.tick_params(which='major', width=2.00)
ax1.tick_params(which='minor', width=2.00)
ax1.xaxis.label.set_fontsize(18)
ax1.xaxis.label.set_weight("bold")
ax1.yaxis.label.set_fontsize(18)
ax1.yaxis.label.set_weight("bold")
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_yticklabels(ax1.get_yticks(), weight='bold', color = "blue")
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax1.set_xlabel("Radius of Curvature (µm)")
plt.show()
plt.close()
