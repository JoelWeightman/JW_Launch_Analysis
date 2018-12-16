# -*- coding: utf-8 -*-
"""
N-Dimensional Machine Learning Genetic Algorithm Adapted for Rocket Trajectory
Joel Weightman

"""


import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.integrate as integ
from multiprocessing import Pool
import Gravity_Turn_Trajectory_ML_shooting as GT
import Gravity_Turn_Trajectory_ML_shooting_bell as GT_bell
import sys
import io
import scipy.optimize as optim

def get_input_variables(m_dry_max, event_alt_max, GT_angle_max, alpha_max):
    
    input_variables = np.array([0.5,0.5,0.5,0.0,0.0,0.5])
    
    stage_delta_vee_ratios = np.array([input_variables[0],input_variables[1]])*9 + 1 ### MASS RATIOS
    event_alt = input_variables[2]*event_alt_max + 1e3
    accel_init = input_variables[3:5]*1 + 0.5 ## at least 1 gee of accel
    alpha = alpha_max*input_variables[5]
    
    m_dry = m_dry_max #input_variables[0]
    GT_angle = 89*np.pi/180 #np.pi/2 - input_variables[2]*GT_angle_max*np.pi/180
    stage_mass_ratios = np.array([0.8,0.2])    
    
    return m_dry, event_alt, GT_angle, stage_mass_ratios, stage_delta_vee_ratios, alpha, accel_init

def calculate_result(engine, weights, m_dry_max, event_alt_max, alpha_max, GT_angle_max, refine):
    
    basin = True
            
    m_dry, event_alt, GT_angle, stage_mass_ratios, stage_delta_vee_ratios, alpha, accel_init = get_input_variables(m_dry_max, event_alt_max, GT_angle_max, alpha_max)
    x_0 = np.array([stage_delta_vee_ratios[0],stage_delta_vee_ratios[1],event_alt,accel_init[0],accel_init[1],alpha])
           
    xmin = [2,2,1e3,0.5,0.5,alpha_max]
    xmax = [10,10,5e3,2,2,0]
    
    # rewrite the bounds in the way required by L-BFGS-B
    bounds = [(low, high) for low, high in zip(xmin, xmax)] #bounds = bounds
    
    # use method L-BFGS-B because the problem is smooth and bounded
    minimizer_kwargs = dict(method="L-BFGS-B", bounds = bounds, args = (m_dry,stage_mass_ratios,GT_angle,weights),tol=1e-2, options = {'maxiter':200,'disp':True})
    
    if basin == True:
        if engine == 'bell':
            pop = optim.basinhopping(GT_bell.run_trajectory, x_0, minimizer_kwargs = minimizer_kwargs)
        else:
            pop = optim.basinhopping(GT.run_trajectory, x_0, minimizer_kwargs = minimizer_kwargs)
    else:
        if engine == 'bell':
            pop = optim.minimize(GT_bell.run_trajectory, x_0, bounds = bounds, args = (m_dry,stage_mass_ratios,GT_angle,weights), method='L-BFGS-B', tol=1e-3, options = {'disp':True})
        else:
            pop = optim.minimize(GT.run_trajectory, x_0, bounds = bounds, args = (m_dry,stage_mass_ratios,GT_angle,weights), method='L-BFGS-B', tol=1e-3, options = {'disp':True})
    
    return pop

def run(engine, W_vel, W_alt, W_angle, W_mass, m_dry_max, event_alt_max, refine, alpha_max, GT_angle_max):
    
#    plt.close("all")
    
    [W_vel, W_alt, W_angle, W_mass] = np.array([W_vel, W_alt, W_angle, W_mass])/(W_vel + W_alt + W_angle + W_mass)
    
    weights = np.array([W_vel, W_alt, W_angle, W_mass])
   
    pop = calculate_result(engine, weights, m_dry_max, event_alt_max, alpha_max, GT_angle_max, refine)
    
    return pop

def n_d_runfile(engine, W_vel, W_alt, W_angle, W_mass, m_dry_max = 1, event_alt_max = 1, refine = False, alpha_max = 0, GT_angle_max = 1):

    pop = run(engine, W_vel, W_alt, W_angle, W_mass, m_dry_max, event_alt_max, refine, alpha_max, GT_angle_max)
        
    filename = 'Result_Files/Population_Results_' + engine + '_%d.npy' % filename_num
    np.save(filename, pop)

    
    return pop

def run_GT_analysis(W_0,optimize_me = True):
    
    [W_vel, W_alt, W_angle, W_mass] = W_0/np.sum(W_0)
    
    plt.close('all')

    already_run = False
    ready_for_refine = False
    ready_for_refine2 = True
    
    global filename_num
    filename_num = 3
    
    engine = 'spike'
          
    m_dry_max = 0.15e3
    event_alt_max = 5e3
    alpha_max = -10 ## In degrees
    alpha_max*= np.pi/180
    GT_angle_max = 2

    start_time = time.time()
    pop = n_d_runfile(engine, W_vel, W_alt, W_angle, W_mass, m_dry_max, event_alt_max, False, alpha_max, GT_angle_max)
    total_time = time.time()-start_time
    print('time = %3.3fs' %total_time)
    
    m_dry, event_alt, GT_angle, stage_mass_ratios, stage_delta_vee_ratios, alpha, accel_init = get_input_variables(m_dry_max, event_alt_max, GT_angle_max, alpha_max)
    
    [stage_delta_vee_ratios_1,stage_delta_vee_ratios_2,event_alt,accel_init_1,accel_init_2,alpha] = pop.x
    stage_delta_vee_ratios = np.array([stage_delta_vee_ratios_1,stage_delta_vee_ratios_2])
    accel_init = np.array([accel_init_1,accel_init_2])
    
    v, phi, r, theta, m, t, ind, delta_vee_grav, score = GT.run_trajectory_final(stage_delta_vee_ratios,m_dry,stage_mass_ratios,event_alt,GT_angle,np.array([0,0,0,0]),alpha,accel_init)

    ind_burn = ind + 2000 - 4
  
    g = 9.80665
    R_e = 6.38e6
    m_dot_1 = -(m[1]-m[0])/(t[1]-t[0])
    m_dot_2 = -(m[2000] - m[1999])/(t[2000]-t[1999])
    thrust_SL = m[0]*(1+accel_init[0])*g
    Isp_SL = thrust_SL/(m_dot_1*g)
    
    D = GT.drag_current(v,0.2,r-R_e,0.5)
    delta_vee_drag = np.sum(D[:2000]/m[:2000]*(t[1:2001] - t[:2000]))
    
    
    phi_mean = abs(np.mean(phi[ind_burn:]))
    phi_std = np.std(phi[ind_burn:])
    
    if phi_mean > 0.1:
        phi_range = phi_mean 
    else:
        phi_range = phi_std
    
    print(phi_range)
    
    if optimize_me == True:
        return phi_range
    else:
        return stage_delta_vee_ratios,event_alt,accel_init,alpha,v, phi, r, theta, m, t, ind, delta_vee_drag, delta_vee_grav, phi_range
    

if __name__ == "__main__":
       
    optimize_weights = False
    
    W_0 = np.array([0.25,0.25,2.0,0.25])
    
    if optimize_weights == True:
        weights = optim.minimize(run_GT_analysis, W_0, method = 'L-BFGS-B', tol=1e-2, options = {'maxiter':100,'disp':True})
        [W_vel, W_alt, W_angle, W_mass] = weights.x
    else:
        stage_delta_vee_ratios,event_alt,accel_init,alpha,v, phi, r, theta, m, t, ind, delta_vee_drag, delta_vee_grav, phi_range = run_GT_analysis(W_0,optimize_me = False)
    
    
    
    