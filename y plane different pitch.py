# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:13:49 2023

@author: limyu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.stats import linregress
import time

pitch = [0.7,0.8,0.9,1.0,1.1,1.2]

t0 = time.time()

for p in pitch:
    t1 = time.time()
    
    # import the electric field distribution on the y-plane monitor as dataframe 
    df = pd.read_excel("C:\\Users\\limyu\\Google Drive\\focusing grating\\2D grating GDS\\grating type 4 branchs\\focusing grating\\2D straight grating duty cycle 0.5 pitch fixed at "+str(p)+" um.xlsx",header=None)
    df1 = df.drop(df.index[0])
    df1 = df1.drop(df1.columns[0], axis=1)
    x = np.linspace(-50e-6, 150e-6, num=6488)
    x = x*1000000
    y = np.linspace(0.0, 95e-6, num=1313)
    y = y*1000000
    X,Y = np.meshgrid(x,y)
# find the maximum value of each row of the dataframe
    row1 = []
    col1 = []
    df1 = df1.reset_index().drop('index', axis=1)
    for index, row in df1.iterrows():
        row_index, col_index = np.where(df1.values == max(row))
        row_index = row_index[0]
        col_index = col_index[0]
        row1.append(row_index)
        col1.append(col_index)
    y1num = [(y[i]) for i in row1]
    y1num = pd.Series(y1num)
    x1num = [(x[i]) for i in col1]
    x1num = pd.Series(x1num) 
# find the slope of the maximum values scatter plot using scipy package    
    slope, intercept, r_value, p_value, std_err = linregress(x1num[222:1150], y1num[222:1150])
# plot the scatter values and linear fitting of the maximum values    
    ax2 = plt.axes()
    ax2.scatter(x1num[222:1150], y1num[222:1150], s=10, alpha=0.2, marker = "s")
    ax2.plot(x1num[222:1150], intercept + slope * x1num[222:1150], 'r')     

        
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    ax2.xaxis.label.set_fontsize(20)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(20)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.legend(["Scattered Data", "Linear fitting"], prop={'weight': 'bold', 'size': 15})
    plt.xlabel("x-position (µm)")
    plt.ylabel("z-position (µm)")
    plt.show()
    plt.close()
    
    
    
    
    df1 = df1.to_numpy()
    colorbarmax = max(df1.max(axis=1))

    fig = plt.figure(figsize=(18, 4))
    ax = plt.axes()
    ax.plot(x1num[222:1150], intercept + slope * x1num[222:1150], color = "white", linestyle='dotted')  
    cp=ax.contourf(X,Y,df1, 200, zdir='z', offset=-100, cmap='hot')
    clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=7), decimals=1)).tolist())
    clb.ax.set_title('Electric Field (eV)', fontweight="bold", fontsize = 20)
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(20)
    ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.set_ylabel('z-position (µm)', fontsize=20, fontweight="bold", labelpad=1)


    ax.xaxis.label.set_fontsize(20)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(20)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(["0", "25", "50", "75", "100", "125", "150", "175", "200" ], weight='bold')

    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.show()
    plt.close()


    t2 = time.time() - t1
    t2 = t2/60
    print("time taken: ",str(p), "um", t2)
    # display the gradient of the line
    print("Gradient: ",str(p), "um", slope)
    angle = np.arctan(slope)
    angle = np.degrees(angle)
    angle = 90 - angle
    print("Angle: ",str(p), "um", angle)
    print("Standard Error: ",str(p), "um", std_err)
    print("r_value: ",str(p), "um", r_value)
    print("p_value: ",str(p), "um", p_value)



print(" ")
t3 = time.time() - t0
t3 = t3/60
print("total time taken: ", t3 )

x = np.linspace(0.0, 200e-6, num=6488)
x = x*1000000
df1 = pd.DataFrame(df1)
ax2 = plt.axes()
ax2.plot(x, df1.iloc[345, :])
ax2.plot(x, df1.iloc[622, :])
ax2.plot(x, df1.iloc[1174, :])
ax2.xaxis.label.set_fontsize(20)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(20)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("x-position (µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["20 µm", "40 µm", "80 µm"], prop={'weight': 'bold', 'size': 15})
plt.show()
plt.close()