# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:12:42 2019

@author: JL
"""
import numpy as np
import glob


## read output parameter file

path = "D:\Rocket\Cold_Flow_Nozzle\CF_Truncation_PR50_Sweep_60_files\dp13\FLU\Fluent\outputs\*.out"

files = glob.glob(path)

results = np.zeros((np.size(files),2))


for i,file in enumerate(files):
    
    splits = file.split('_')
    pr_index = splits.index('PR')
    press_ratio = int(splits[pr_index+1][:-4])/10
    results[i,0] = press_ratio
    
    f = open(file,'r')
    for line in f:
        if 'isp-op' in line:
            isp_line = line
            
    isp = float(isp_line[10:-5])
    results[i,1] = isp
    

results = results[results[:,0].argsort()]


np.save(path[:-5]+'results',results)


#plt.figure()
#plt.plot(results[:,0],results[:,1])
#plt.xlabel('PR')
#plt.ylabel('Isp')