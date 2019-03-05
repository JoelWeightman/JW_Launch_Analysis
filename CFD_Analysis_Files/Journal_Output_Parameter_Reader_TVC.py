# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:12:42 2019

@author: JL
"""
import numpy as np
import glob
import matplotlib.pyplot as plt


## read output parameter file

path = "D:\Rocket\Cold_Flow_Nozzle\CF_TVC_Truncation_PR50_Sweep_60_files\dp13\FLU\Fluent\outputs\*.out"

files = glob.glob(path)

results = np.zeros((np.size(files),6))


for i,file in enumerate(files):
    
    splits = file.split('_')
    pr_index = splits.index('Pressure')
    press_ratio = int(splits[pr_index+1][:-4])
    results[i,0] = press_ratio
    
    f = open(file,'r')
    for line in f:
        if 'outlet-op' in line:
            m_out_line = line
        if 'transverse_force-op' in line:
            y_force_line = line
        if 'axial_thrust-op' in line:
            x_force_line = line
            
    m_out = float(m_out_line[15:-22])
    x_force = float(x_force_line[15:-22])
    y_force = float(y_force_line[19:-22])
    results[i,1] = m_out
    results[i,2] = x_force
    results[i,3] = y_force
    

results[:,4] = results[:,2]/(results[:,1]*9.81)
results[:,5] = results[:,3]/(results[:,1]*9.81)

results = results[results[:,0].argsort()]


np.save(path[:-5]+'results',results)

plt.figure()
plt.plot(results[:,0]/500000*100,results[:,4])
plt.plot(results[:,0]/500000*100,results[:,5])
plt.xlabel('Inlet 2 Throttle %')
plt.ylabel('Isp')