# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:35:18 2019

@author: JL
"""

import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as cp

## Load in CFD results
results_aero_80_300_orig = np.load("D:/Rocket/Cold_Flow_Nozzle/CF_Truncation_PR50_Sweep_80_files/dp13/FLU/Fluent/outputs/results.npy")
pressures_aero_80_300_orig = results_aero_80_300_orig[:,0]
Isp_aero_80_300_orig = results_aero_80_300_orig[:,1]

results_aero_80_300_low = np.load("D:/Rocket/Hot_Flow/Hot_Flow_80_Perc_Trunc_Higher_Res_files/dp13/FLU/Fluent/outputs/results_300.npy")
pressures_aero_80_300_low = results_aero_80_300_low[:,0]
Isp_aero_80_300_low = results_aero_80_300_low[:,1]

results_aero_80_650_low = np.load("D:/Rocket/Hot_Flow/Hot_Flow_80_Perc_Trunc_Higher_Res_files/dp13/FLU/Fluent/outputs/results_650.npy")
pressures_aero_80_650_low = results_aero_80_650_low[:,0]
Isp_aero_80_650_low = results_aero_80_650_low[:,1]

results_aero_80_1050_low = np.load("D:/Rocket/Hot_Flow/Hot_Flow_80_Perc_Trunc_Higher_Res_files/dp13/FLU/Fluent/outputs/results_1050.npy")
pressures_aero_80_1050_low = results_aero_80_1050_low[:,0]
Isp_aero_80_1050_low = results_aero_80_1050_low[:,1]


## Align pressure ratios for comparison
temp_1,temp_2,orig_indices = np.intersect1d(pressures_aero_80_300_low,pressures_aero_80_300_orig,return_indices=True)
pressures_aero_80_300_orig = pressures_aero_80_300_orig[orig_indices]
Isp_aero_80_300_orig = Isp_aero_80_300_orig[orig_indices]

pressures_aero_80_3850_low = np.array([50])
Isp_aero_80_3850_low = np.array([236.645])


## Check error between orig CFD and new low res
error_perc_300_orig_low = np.abs((Isp_aero_80_300_orig-Isp_aero_80_300_low)/Isp_aero_80_300_orig)*100


## Scaling factor between 300K CFD and XK CFD
scaling_factor_300_650_low = np.abs((Isp_aero_80_650_low)/Isp_aero_80_300_low)
scaling_factor_300_1050_low = np.abs((Isp_aero_80_1050_low)/Isp_aero_80_300_low)

scaling_factor_300_3850_low_mean = np.abs((Isp_aero_80_3850_low)/Isp_aero_80_300_low[3])[0]
scaling_factor_300_650_low_mean = np.mean(scaling_factor_300_650_low)
scaling_factor_300_1050_low_mean = np.mean(scaling_factor_300_1050_low)


# Figure of scaling factor vs pressure ratio
plt.figure()
plt.plot(pressures_aero_80_650_low,scaling_factor_300_650_low,'g-')
plt.plot(pressures_aero_80_1050_low,scaling_factor_300_1050_low,'r-')


## trying to find correct scaling parameter
air_rho_300 = cp.PropsSI('D','T',300,'P',5e5,'air')
air_rho_650 = cp.PropsSI('D','T',650,'P',5e5,'air')
air_rho_1050 = cp.PropsSI('D','T',1050,'P',5e5,'air')
air_rho_3850 = cp.PropsSI('D','T',3850,'P',5e5,'air')

air_gamma_300 = cp.PropsSI('CP0MASS','T',300,'P',5e5,'air')/cp.PropsSI('CVMASS','T',300,'P',5e5,'air')
air_gamma_650 = cp.PropsSI('CP0MASS','T',650,'P',5e5,'air')/cp.PropsSI('CVMASS','T',650,'P',5e5,'air')
air_gamma_1050 = cp.PropsSI('CP0MASS','T',1050,'P',5e5,'air')/cp.PropsSI('CVMASS','T',1050,'P',5e5,'air')
air_gamma_3850 = cp.PropsSI('CP0MASS','T',3850,'P',5e5,'air')/cp.PropsSI('CVMASS','T',3850,'P',5e5,'air')

rho_ratio_300_650 = air_rho_300/air_rho_650
rho_ratio_300_1050 = air_rho_300/air_rho_1050
rho_ratio_300_3850 = air_rho_300/air_rho_3850

sqrt_rho_ratio_300_650 = np.sqrt(rho_ratio_300_650)
sqrt_rho_ratio_300_1050 = np.sqrt(rho_ratio_300_1050)
sqrt_rho_ratio_300_3850 = np.sqrt(rho_ratio_300_3850)


##Scaling parameter error
scaling_error_300_650 = (scaling_factor_300_650_low_mean-sqrt_rho_ratio_300_650)/scaling_factor_300_650_low_mean*100
scaling_error_300_1050 = (scaling_factor_300_1050_low_mean-sqrt_rho_ratio_300_1050)/scaling_factor_300_1050_low_mean*100
scaling_error_300_3850 = (scaling_factor_300_3850_low_mean-sqrt_rho_ratio_300_3850)/scaling_factor_300_3850_low_mean*100


## Scale up orig 300K CFD to 3850K based on scaling parameter
results_aero_80_300_orig = np.load("D:/Rocket/Cold_Flow_Nozzle/CF_Truncation_PR50_Sweep_80_files/dp13/FLU/Fluent/outputs/results.npy")
pressures_aero_80_300_orig_temp = results_aero_80_300_orig[:,0]
Isp_aero_80_300_orig_temp = results_aero_80_300_orig[:,1]
pressures_aero_80_3850_scaled = pressures_aero_80_300_orig_temp
Isp_aero_80_3850_scaled = Isp_aero_80_300_orig_temp*np.sqrt(cp.PropsSI('D','T',300,'P',5e5,'air')/cp.PropsSI('D','T',3850,'P',5e5,'air'))


## Plot
plt.figure()
plt.plot(pressures_aero_80_300_orig,Isp_aero_80_300_orig,'ko',pressures_aero_80_300_orig,Isp_aero_80_300_orig,'k-')
plt.plot(pressures_aero_80_300_low,Isp_aero_80_300_low,'bo',pressures_aero_80_300_low,Isp_aero_80_300_low,'b-')
plt.plot(pressures_aero_80_650_low,Isp_aero_80_650_low,'go',pressures_aero_80_650_low,Isp_aero_80_650_low,'g-')
plt.plot(pressures_aero_80_1050_low,Isp_aero_80_1050_low,'ro',pressures_aero_80_1050_low,Isp_aero_80_1050_low,'r-')
plt.plot(pressures_aero_80_3850_scaled,Isp_aero_80_3850_scaled,'ko',pressures_aero_80_3850_scaled,Isp_aero_80_3850_scaled,'k-')
plt.xlim([0,210])
