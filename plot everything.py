# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 17:21:49 2023

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:05:02 2023

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

t0 = time.time()

master_data_horizontal_3d = np.empty((0, 10000, 7))
master_data_horizontal_x_3d = np.empty((0, 10000, 7))
horizontal_peaks_df =pd.DataFrame([])
horizontal_peaks_max_df = pd.DataFrame([])
horizontal_peaks_position_df = pd.DataFrame([])
horizontal_half_df = pd.DataFrame([])
horizontal_full_df =pd.DataFrame([])
full_width_horizontal_df= pd.DataFrame([])
half_width_horizontal_df= pd.DataFrame([])
horizontal_cut_df = pd.DataFrame([])


master_data_verticle_3d = np.empty((0, 10000, 7))
verticle_peaks_df =pd.DataFrame([])
verticle_peaks_max_df = pd.DataFrame([])
verticle_peaks_position_df = pd.DataFrame([])
verticle_half_df = pd.DataFrame([])
verticle_full_df =pd.DataFrame([])
full_width_verticle_df= pd.DataFrame([])
half_width_verticle_df= pd.DataFrame([])
verticle_cut_df = pd.DataFrame([])


radius = ["15um", "20um", "30um", "40um", "50um", "60um"]


for r in radius:
    master_data_horizontal = pd.DataFrame([])
    master_data_horizontal_x = pd.DataFrame([])
    horizontal_peaks = []
    horizontal_peaks_position = []
    horizontal_peaks_max = []
    horizontal_half = []
    horizontal_full = []
    horizontal_cut = []
    horizontal_level = [0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01]

    
    master_data_verticle = pd.DataFrame([])
    verticle_peaks = []
    verticle_peaks_position = []
    verticle_peaks_max = []
    verticle_half = []
    verticle_full = []
    verticle_cut = []
    verticle_level = [0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01]
    z_axis_height = [1,2,3,4,5,6,7]
    for i, h in enumerate(z_axis_height):
        print(" ")
        print(" ")
        print(r)
        print(str(h))
        print("C:\\Users\\limyu\\Google Drive\\focusing grating\\2D grating GDS\\grating type 4 branchs\\3D simulation result\\"+str(h)+"e-05grating012umpitch05dutycycle"+r+".xlsx")
        df = pd.read_excel("C:\\Users\\limyu\\Google Drive\\focusing grating\\2D grating GDS\\grating type 4 branchs\\3D simulation result\\" +str(h)+ "e-05grating012umpitch05dutycycle"+r+".xlsx",header=None)
        df = df.drop(df.index[0])

            

        x = np.linspace(0.0, 180e-6, num=2113)
        x = x*1000000
        y = np.linspace(0.0, 40e-6, num=781)
        y = y*1000000
        


        X,Y = np.meshgrid(x,y)
        df1 = df.to_numpy()
        colorbarmax = max(df1.max(axis=1))
        
        max_df = max(df1.max(axis=1))
        columns = df.columns[df.eq(max_df).any()].tolist()
            
        print(max_df)
        print(columns)
        print(int(float(columns[0])))

        fig = plt.figure(figsize=(18, 4))
        ax = plt.axes()
        cp=ax.contourf(X,Y,df1, 200, zdir='z', offset=-100, cmap='hot')
        clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
        clb.ax.set_title('Electric Field (eV)', fontweight="bold")
        for l in clb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(15)
        ax.set_xlabel('x-position (µm)', fontsize=15, fontweight="bold", labelpad=1)
        ax.set_ylabel('y-position (µm)', fontsize=15, fontweight="bold", labelpad=1)


        ax.xaxis.label.set_fontsize(15)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_fontsize(15)
        ax.yaxis.label.set_weight("bold")
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_yticklabels(ax.get_yticks(), weight='bold')
        ax.set_xticklabels(ax.get_xticks(), weight='bold')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.show()
        plt.close()
            
            
        #plot horizontal
        df_hor = df.iloc[380,:]
        
        max_index = np.argmax(df_hor)
        df_hor_slice = df_hor[(max_index - 390):(max_index + 390)]
        df_hor_slice = df_hor_slice.reset_index(drop=True)
        x_slice = x[(max_index - 390):(max_index + 390)]
        ax2 = plt.axes()
        tck = interpolate.splrep(x_slice, df_hor_slice.tolist(), s=0.0005, k=4) 
        x_new = np.linspace(min(x_slice), max(x_slice), 10000)
        y_fit = interpolate.BSpline(*tck)(x_new)
        peaks, _ = find_peaks(y_fit)
        peaks_h = x_new[peaks]
            
        
        horizontal_peaks.append(peaks_h)
        
        horizontal_peaks_position.append(x_new[np.where(y_fit == max(y_fit))[0][0]])
        
        horizontal_peaks_max.append(df.max().max())

        results_half = peak_widths(y_fit, peaks, rel_height=0.5)
        width = results_half[0]
        width = [i*(x_new[1] - x_new[0]) for i in width]
        width = np.array(width)
        height = results_half[1]
        x_min = results_half[2]
        x_min = np.array(x_new[np.around(x_min, decimals=0).astype(int)])
        x_max = results_half[3]
        x_max = np.array(x_new[np.around(x_max, decimals=0).astype(int)])    
        results_half_plot = (width, height, x_min, x_max)

        results_full = peak_widths(y_fit, peaks, rel_height=0.865)
        width_f = results_full[0]
        width_f = [i*(x_new[1] - x_new[0]) for i in width_f]
        width_f = np.array(width_f)
        height_f = results_full[1]
        x_min_f = results_full[2]
        x_min_f = np.array(x_new[np.around(x_min_f, decimals=0).astype(int)])
        x_max_f = results_full[3]
        x_max_f = np.array(x_new[np.around(x_max_f, decimals=0).astype(int)]) 
        results_full_plot = (width_f, height_f, x_min_f, x_max_f)
            
        
        horizontal_half.append(max(results_half_plot[0]))      
        horizontal_full.append(max(results_full_plot[0]))
        
        #Determine the cross-section at y = y_line
        y_line = horizontal_level[i]
        delta = y_line/10
        x_close = x_new[np.where(np.abs(y_fit - y_line) < delta)]
        peak_width = np.max(x_close) - np.min(x_close)
        horizontal_cut.append(peak_width)        

            
        ax2.plot(x_new, y_fit)
        ax2.plot(peaks_h, y_fit[peaks], "o")
        ax2.hlines(*results_half_plot[1:], color="C2")
        ax2.hlines(*results_full_plot[1:], color="C3")
        ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")

            
        ax2.tick_params(which='major', width=2.00)
        ax2.tick_params(which='minor', width=2.00)

        ax2.xaxis.label.set_fontsize(15)
        ax2.xaxis.label.set_weight("bold")
        ax2.yaxis.label.set_fontsize(15)
        ax2.yaxis.label.set_weight("bold")
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
        ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
        ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines['bottom'].set_linewidth(2)
        ax2.spines['left'].set_linewidth(2)
        plt.xlabel("x-position (µm)")
        plt.ylabel("Electric Field (eV)")
        plt.legend(["Electric Field (eV)", "Peaks", "FWHM", "Full Width", "E = "+str(horizontal_level[i])+" eV"], prop={'weight': 'bold','size': 10})
        plt.show()
        plt.close()

        ax2 = plt.axes()
        ax2.plot(x_new, y_fit)
        ax2.plot(peaks_h, y_fit[peaks], "o")
        ax2.hlines(*results_half_plot[1:], color="C2")
        ax2.hlines(*results_full_plot[1:], color="C3")          
        ax2.tick_params(which='major', width=2.00)
        ax2.tick_params(which='minor', width=2.00)
        ax2.xaxis.label.set_fontsize(15)
        ax2.xaxis.label.set_weight("bold")
        ax2.yaxis.label.set_fontsize(15)
        ax2.yaxis.label.set_weight("bold")
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
        ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
        ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines['bottom'].set_linewidth(2)
        ax2.spines['left'].set_linewidth(2)
        plt.xlabel("x-position (µm)")
        plt.ylabel("Electric Field (eV)")
        plt.legend(["Electric Field (eV)", "Peaks", "FWHM", "Full Width"], prop={'weight': 'bold','size': 10})
        plt.show()
        plt.close()

            
        master_data_horizontal_x[str(h)] = x_new

        master_data_horizontal[str(h)] = y_fit
            
        #plot verticle
        df_ver = df.iloc[:,int(float(columns[0]))]
        ax2 = plt.axes()
        tck = interpolate.splrep(y, df_ver.tolist(), s=0, k=4) 
        x_new = np.linspace(min(y), max(y), 10000)
        y_fit = interpolate.BSpline(*tck)(x_new)
        peaks, _ = find_peaks(y_fit)
        peaks_v = x_new[peaks]
            
        
        verticle_peaks.append(peaks_v)
        
        verticle_peaks_position.append(x_new[np.where(y_fit == max(y_fit))[0][0]])
        
        verticle_peaks_max.append(df.max().max())

        results_half = peak_widths(y_fit, peaks, rel_height=0.5)
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

        results_full = peak_widths(y_fit, peaks, rel_height=0.865)
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

        #Determine the cross-section at y = y_line
        y_line = verticle_level[i]
        delta = y_line/10
        x_close = x_new[np.where(np.abs(y_fit - y_line) < delta)]
        peak_width = np.max(x_close) - np.min(x_close)
        verticle_cut.append(peak_width)     
        
        
        ax2.plot(x_new, y_fit)
        ax2.plot(peaks_v, y_fit[peaks], "o")
        ax2.hlines(*results_half_plot[1:], color="C2")
        ax2.hlines(*results_full_plot[1:], color="C3")
        ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")

            
        ax2.tick_params(which='major', width=2.00)
        ax2.tick_params(which='minor', width=2.00)
        
        ax2.xaxis.label.set_fontsize(15)
        ax2.xaxis.label.set_weight("bold")
        ax2.yaxis.label.set_fontsize(15)
        ax2.yaxis.label.set_weight("bold")
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
        ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
        ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines['bottom'].set_linewidth(2)
        ax2.spines['left'].set_linewidth(2)
        plt.xlabel("y-position (µm)")
        plt.ylabel("Electric Field (eV)")
        plt.legend(["Electric Field (eV)", "Peaks", "FWHM", "Full Width", "E = "+str(verticle_level[i])+" eV"], prop={'weight': 'bold','size': 10})
        plt.show()
        plt.close()
        
        
        ax2 = plt.axes()
        ax2.plot(x_new, y_fit)
        ax2.plot(peaks_v, y_fit[peaks], "o")
        ax2.hlines(*results_half_plot[1:], color="C2")
        ax2.hlines(*results_full_plot[1:], color="C3")
        ax2.hlines(y = y_line, xmin = np.min(x_close), xmax = np.max(x_close), color = "C4")            
        ax2.tick_params(which='major', width=2.00)
        ax2.tick_params(which='minor', width=2.00)
        ax2.xaxis.label.set_fontsize(15)
        ax2.xaxis.label.set_weight("bold")
        ax2.yaxis.label.set_fontsize(15)
        ax2.yaxis.label.set_weight("bold")
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
        ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
        ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines['bottom'].set_linewidth(2)
        ax2.spines['left'].set_linewidth(2)
        plt.xlabel("y-position (µm)")
        plt.ylabel("Electric Field (eV)")
        plt.legend(["Electric Field (eV)", "Peaks", "FWHM", "Full Width"], prop={'weight': 'bold','size': 10})
        plt.show()
        plt.close()

        master_data_verticle[str(h)] = y_fit
    master_data_horizontal = master_data_horizontal.values
    master_data_horizontal_3d = np.concatenate((master_data_horizontal_3d, np.expand_dims(master_data_horizontal, axis=0)), axis=0)
    master_data_horizontal = pd.DataFrame(master_data_horizontal)
    master_data_horizontal_x = master_data_horizontal_x.values
    master_data_horizontal_x_3d = np.concatenate((master_data_horizontal_x_3d, np.expand_dims(master_data_horizontal_x, axis=0)), axis=0)
    master_data_horizontal_x = pd.DataFrame(master_data_horizontal_x)
    horizontal_peaks = pd.Series(horizontal_peaks)
    horizontal_peaks_df[r] = horizontal_peaks
    horizontal_peaks_position = pd.Series(horizontal_peaks_position)
    horizontal_peaks_position_df[r] = horizontal_peaks_position
    horizontal_peaks_max = pd.Series(horizontal_peaks_max)
    horizontal_peaks_max_df[r] = horizontal_peaks_max    
    horizontal_half = pd.Series(horizontal_half)
    horizontal_half_df[r] = horizontal_half 
    horizontal_full = pd.Series(horizontal_full)
    horizontal_full_df[r] = horizontal_full 
    horizontal_cut_df[r] = horizontal_cut
    
    master_data_verticle = master_data_verticle.values
    master_data_verticle_3d = np.concatenate((master_data_verticle_3d, np.expand_dims(master_data_verticle, axis=0)), axis=0)
    master_data_verticle = pd.DataFrame(master_data_verticle)
    verticle_peaks = pd.Series(verticle_peaks)
    verticle_peaks_df[r] = verticle_peaks
    verticle_peaks_position = pd.Series(verticle_peaks_position)
    verticle_peaks_position_df[r] = verticle_peaks_position
    verticle_peaks_max = pd.Series(verticle_peaks_max)
    verticle_peaks_max_df[r] = verticle_peaks_max    
    verticle_half = pd.Series(verticle_half)
    verticle_half_df[r] = verticle_half 
    verticle_full = pd.Series(verticle_full)
    verticle_full_df[r] = verticle_full 
    verticle_cut_df[r] = verticle_cut




    #plot horizontal all
    ax2 = plt.axes()
    for col1, col2 in zip(master_data_horizontal_x.columns, master_data_horizontal.columns): 
        ax2.plot(master_data_horizontal_x[col1], master_data_horizontal[col2])
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)

    ax2.xaxis.label.set_fontsize(15)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(15)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("x-position (µm)")
    plt.ylabel("Electric Field (eV)")
    plt.legend(["10µm", "20µm", "30µm", "40µm", "50µm", "60µm", "70µm"], prop={'weight': 'bold', 'size': 10})
    plt.show()
    plt.close()

    #plot vertical all
    ax2 = plt.axes()
    x_new = np.linspace(min(y), max(y), 10000)
    for col in master_data_verticle.columns: 
        ax2.plot(x_new, master_data_verticle[col])
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)

    ax2.xaxis.label.set_fontsize(15)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(15)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) 
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("y-position (µm)")
    plt.ylabel("Electric Field (eV)")
    plt.legend(["10µm", "20µm", "30µm", "40µm", "50µm", "60µm", "70µm"], prop={'weight': 'bold', 'size': 10})
    plt.show()
    plt.close()


#plot gradient
height_micron = [10, 20, 30, 40, 50, 60, 70]

ax2 = plt.axes()
for col in horizontal_peaks_position_df.columns:
    slope, intercept, _, _, _ = linregress(height_micron, horizontal_peaks_position_df[col])
    ax2.scatter(height_micron, horizontal_peaks_position_df[col])
    y_slope = [(i*slope+intercept) for i in height_micron]
    ax2.plot(height_micron, y_slope, 'r')
    print(slope)
    angle = np.arctan(slope) * 180 / np.pi
    print(angle)
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(15)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(15)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("Height (µm)")
plt.ylabel("z-position (µm)")
plt.legend(["Peaks", "Linear Fit"], prop={'weight': 'bold'})
plt.show()
plt.close()


horizontal_list = [horizontal_peaks_max_df, horizontal_peaks_position_df, 
                   horizontal_half_df, horizontal_full_df]

verticle_list = [verticle_peaks_max_df, verticle_peaks_position_df, 
                   verticle_half_df, verticle_full_df]

marker_list = ['o', 'v', 's', 'p', "*", "^"]
height_micron = [10, 20, 30, 40, 50, 60, 70]
ax2 = plt.axes()

print("horizontal_peaks_max_df")
for j, column in enumerate(horizontal_peaks_max_df.columns):        
    ax2.plot(height_micron, horizontal_peaks_max_df[column], marker=marker_list[j], linestyle='-')
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(15)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(15)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("Height (µm)")
plt.ylabel("Electric Field (eV)")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold', 'size': 10})
plt.show()
plt.close()

ax2 = plt.axes()
print("horizontal_peaks_position_df")
for j, column in enumerate(horizontal_peaks_position_df.columns):        
    ax2.plot(height_micron, horizontal_peaks_position_df[column], marker=marker_list[j], linestyle='-')
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(15)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(15)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("Height (µm)")
plt.ylabel("x-position (µm)")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold', 'size': 10})
plt.show()
plt.close()

ax2 = plt.axes()
print("horizontal_half_df")
for j, column in enumerate(horizontal_half_df.columns):        
    ax2.plot(height_micron, horizontal_half_df[column], marker=marker_list[j], linestyle='-')
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(15)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(15)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("Height (µm)")
plt.ylabel("FWHM (µm)")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold', 'size': 10})
plt.show()
plt.close()

ax2 = plt.axes()
print("verticle_half_df")
for j, column in enumerate(verticle_half_df.columns):        
    ax2.plot(height_micron, verticle_half_df[column], marker=marker_list[j], linestyle='-')
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(15)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(15)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("Height (µm)")
plt.ylabel("FWHM (µm)")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold', 'size': 10})
plt.show()
plt.close()


ax2 = plt.axes()
print("horizontal_full_df")
for j, column in enumerate(horizontal_full_df.columns):        
    ax2.plot(height_micron, horizontal_full_df[column], marker=marker_list[j], linestyle='-')
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(15)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(15)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("Height (µm)")
plt.ylabel("FWHM (µm)")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold', 'size': 10})
plt.show()
plt.close()

ax2 = plt.axes()
print("verticle_full_df")
for j, column in enumerate(verticle_full_df.columns):        
    ax2.plot(height_micron, verticle_full_df[column], marker=marker_list[j], linestyle='-')
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(15)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(15)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("Height (µm)")
plt.ylabel("Peak Width (µm)")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold', 'size': 10})
plt.show()
plt.close()

ax2 = plt.axes()
print("horizontal_cut_df")
for j, column in enumerate(horizontal_cut_df.columns):        
    ax2.plot(height_micron, horizontal_cut_df[column], marker=marker_list[j], linestyle='-')
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(15)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(15)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("Height (µm)")
plt.ylabel("Width (µm)")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold', 'size': 10})
plt.show()
plt.close()

ax2 = plt.axes()
print("verticle_cut_df")
for j, column in enumerate(verticle_cut_df.columns):        
    ax2.plot(height_micron, verticle_cut_df[column], marker=marker_list[j], linestyle='-')
ax2.tick_params(which='major', width=2.00)
ax2.tick_params(which='minor', width=2.00)
ax2.xaxis.label.set_fontsize(15)
ax2.xaxis.label.set_weight("bold")
ax2.yaxis.label.set_fontsize(15)
ax2.yaxis.label.set_weight("bold")
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.xlabel("Height (µm)")
plt.ylabel("Width (µm)")
plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold', 'size': 10})
plt.show()
plt.close()

print(" ")
print("=================================Separator=================================")
print(" ")

ax2 = plt.axes()
for i, verticle in enumerate(verticle_list):
    print(f"DataFrame {i}:")
    height_micron = [10, 20, 30, 40, 50, 60, 70]
    ax2 = plt.axes()
    marker_list = ['o', 'v', 's', 'p', "*", "^"]
    for j, column in enumerate(verticle.columns):        
        ax2.plot(height_micron, verticle[column], marker=marker_list[j], linestyle='-')
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    ax2.xaxis.label.set_fontsize(15)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(15)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("Height (µm)")
    plt.ylabel("FWHM (µm)")
    plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold', 'size': 10})
    plt.show()
    plt.close()

ax2 = plt.axes()    
for i, horizontal in enumerate(horizontal_list):
    print(f"DataFrame {i}:")
    height_micron = [10, 20, 30, 40, 50, 60, 70]
    ax2 = plt.axes()
    marker_list = ['o', 'v', 's', 'p', "*", "^"]
    for j, column in enumerate(horizontal.columns):        
        ax2.plot(height_micron, horizontal[column], marker=marker_list[j], linestyle='-')
    ax2.tick_params(which='major', width=2.00)
    ax2.tick_params(which='minor', width=2.00)
    ax2.xaxis.label.set_fontsize(15)
    ax2.xaxis.label.set_weight("bold")
    ax2.yaxis.label.set_fontsize(15)
    ax2.yaxis.label.set_weight("bold")
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticklabels(ax2.get_yticks(), weight='bold')
    ax2.set_xticklabels(ax2.get_xticks(), weight='bold')
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    plt.xlabel("Height (µm)")
    plt.ylabel("FWHM (µm)")
    plt.legend(["15µm", "20µm", "30µm", "40µm", "50µm", "60µm"], prop={'weight': 'bold', 'size': 10})
    plt.show()
    plt.close()

t3 = time.time() - t0
t3 = t3/60
print("Total time taken is "+str(t3)+" minutes")

import seaborn as sns
sns.set(font_scale=1)
height_plot = ["10µm", "20µm", "30µm", "40µm", "50µm", "60µm", "70µm"]

print("verticle_half_df")
verticle_half_df.index = height_plot
# Create a subplot for the heatmap
fig, ax = plt.subplots()
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
# Plot the heatmap with annotations on the subplot
sns.heatmap(verticle_half_df, cmap='hot', annot=True, fmt=".2f", annot_kws={"weight": "bold"}, ax=ax)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("FWHM (µm)", fontweight="bold")
# Set the title of the plot
ax.set_title("FWHM along y-axis (µm)", fontweight = "bold")
# Set the x and y labels of the plot
ax.set_xlabel("Radius of Curvatures", fontweight = "bold")
ax.set_ylabel("Height (along z-axis)", fontweight = "bold")
# Display the plot
plt.show()
plt.close()

print("verticle_peaks_max_df")
verticle_peaks_max_df.index = height_plot
# Create a subplot for the heatmap
fig, ax = plt.subplots()
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
# Plot the heatmap with annotations on the subplot
sns.heatmap(verticle_peaks_max_df, cmap='hot', annot=True, fmt=".3f", annot_kws={"weight": "bold"}, ax=ax)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Electric Field (eV)", fontweight="bold")
# Set the title of the plot
ax.set_title("Peak E-field (eV)", fontweight = "bold")
# Set the x and y labels of the plot
ax.set_xlabel("Radius of Curvatures", fontweight = "bold")
ax.set_ylabel("Height (along z-axis)", fontweight = "bold")
# Display the plot
plt.show()
plt.close()

print("verticle_cut_df")
verticle_cut_df.index = height_plot
# Create a subplot for the heatmap
fig, ax = plt.subplots()
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
# Plot the heatmap with annotations on the subplot
sns.heatmap(verticle_cut_df, cmap='hot', annot=True, fmt=".2f", annot_kws={"weight": "bold"}, ax=ax)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Peak Width (µm)", fontweight="bold")
# Set the title of the plot
ax.set_title("Peak Width along y-axis (µm)", fontweight = "bold")
# Set the x and y labels of the plot
ax.set_xlabel("Radius of Curvatures", fontweight = "bold")
ax.set_ylabel("Height (along z-axis)", fontweight = "bold")
# Display the plot
plt.show()
plt.close()

print("horizontal_half_df")
horizontal_half_df.index = height_plot
# Create a subplot for the heatmap
fig, ax = plt.subplots()
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
# Plot the heatmap with annotations on the subplot
sns.heatmap(horizontal_half_df, cmap='hot', annot=True, fmt=".2f", annot_kws={"weight": "bold"}, ax=ax)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("FWHM (µm)", fontweight="bold")
# Set the title of the plot
ax.set_title("FWHM along x-axis (µm)", fontweight = "bold")
# Set the x and y labels of the plot
ax.set_xlabel("Radius of Curvatures", fontweight = "bold")
ax.set_ylabel("Height (along z-axis)", fontweight = "bold")
# Display the plot
plt.show()
plt.close()

print("horizontal_full_df")
horizontal_full_df.index = height_plot
# Create a subplot for the heatmap
fig, ax = plt.subplots()
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
# Plot the heatmap with annotations on the subplot
sns.heatmap(horizontal_full_df, cmap='hot', annot=True, fmt=".2f", annot_kws={"weight": "bold"}, ax=ax)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Peak Width (µm)", fontweight="bold")
# Set the title of the plot
ax.set_title("Peak Width along x-axis (µm)", fontweight = "bold")
# Set the x and y labels of the plot
ax.set_xlabel("Radius of Curvatures", fontweight = "bold")
ax.set_ylabel("Height (along z-axis)", fontweight = "bold")
# Display the plot
plt.show()
plt.close()




