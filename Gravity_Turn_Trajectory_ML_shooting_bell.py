# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:57:53 2018

@author: JL
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.optimize as opt

def area_ratio_calc(P1,P2,gamma,T1):
    
    M = Mach_number(P1,P2,gamma)
    
    T = T1*(1+(gamma-1)/2*M**2)**(-1)
    A_R = ((gamma+1)/2)**(-(gamma+1)/(2*(gamma-1)))*((1+(gamma-1)/2*M**2)**((gamma+1)/(2*(gamma-1)))/M)

    return A_R, M, T

def exit_area(A_ratio,d_throat):
    
    A_throat = (d_throat/2)**2*np.pi
    
    A_exit = A_throat*A_ratio
    
    return A_exit

def exhaust_velocity(gamma,R,T,Mj):
    
    v_e = Mj*np.sqrt(gamma*R*T)
    
    return v_e

def Mach_number(P1,P2,gamma):
    
    M = np.sqrt(((P1/P2)**((gamma-1)/gamma)-1)*2/(gamma-1))
    
    return M

def current_pressure(alt,g,R):
    
    h = np.array([0,11e3,20e3,32e3,47e3,51e3,71e3])
    P = np.array([101325.0,22632.1,5474.89,868.02,110.91,66.94,3.96])
    T = np.array([288.15,216.65,216.65,228.65,270.65,270.65,214.65])
    L = np.array([-0.0065,0,0.001,0.0028,0,-0.0028,-0.002])
    
    if np.isnan(alt):
        P_var = 0.001
        return P_var
    else:
        
        if alt > 150e3:
            alt = 150e3
        elif alt < 0:
            alt = 0
        
        ind = np.where(abs(alt-h) == np.min(abs(alt-h)))[0][0]
        if alt < h[ind]:
            ind -= 1
    
        if L[ind] == 0:
            P_var = P[ind]*np.exp(-g*(alt-h[ind])/(R*T[ind]))
            T_var = T[ind]
        else:
            P_var = P[ind]*(T[ind]/(T[ind]+L[ind]*(alt-h[ind])))**(g/(R*L[ind]))
            T_var = T[ind]+L[ind]*(alt-h[ind])
            
        rho_var = P_var/(R*T_var)
        return P_var, T_var, rho_var

def thrust_current(alt,alt_des,m_dot,P_chamber,T_chamber,gamma,R_air,R_prop,g,A_exit):
    
    P_alt,T_alt,rho_alt = current_pressure(alt,g,R_air)
    P_des,T_des,rho_des = current_pressure(alt_des,g,R_air)
    
    AR, M, T = area_ratio_calc(P_chamber,P_alt,gamma,T_chamber)
    
    v_e = exhaust_velocity(gamma,R_prop,T,M)
    
    thrust = m_dot*v_e + A_exit*(P_des-P_alt)

    return thrust

def drag_current(V, Cd, d, rocket_radius):

    Area = rocket_radius**2*np.pi
    rho_curr = density_calc(d)
    drag = 0.5 * rho_curr * V * V * Cd * Area
#    print(V)
    return drag


def density_calc(h):

    Temporary_rho = np.array([
        0.00000, 1.22500, 0.100000, 1.21328, 0.200000, 1.20165, 0.300000,
        1.19011, 0.400000, 1.17864, 0.500000, 1.16727, 0.600000, 1.15598,
        0.700000, 1.14477, 0.800000, 1.13364, 0.900000, 1.12260, 1.00000,
        1.11164, 1.10000, 1.10077, 1.20000, 1.08997, 1.30000, 1.07925, 1.40000,
        1.06862, 1.50000, 1.05807, 1.60000, 1.04759, 1.70000, 1.03720, 1.80000,
        1.02688, 1.90000, 1.01665, 2.00000, 1.00649, 2.10000, 0.996410,
        2.20000, 0.986407, 2.30000, 0.976481, 2.40000, 0.966632, 2.50000,
        0.956859, 2.60000, 0.947162, 2.70000, 0.937540, 2.80000, 0.927993,
        2.90000, 0.918520, 3.00000, 0.909122, 3.10000, 0.899798, 3.20000,
        0.890546, 3.30000, 0.881368, 3.40000, 0.872262, 3.50000, 0.863229,
        3.60000, 0.854267, 3.70000, 0.845377, 3.80000, 0.836557, 3.90000,
        0.827808, 4.00000, 0.819129, 4.10000, 0.810520, 4.20000, 0.801981,
        4.30000, 0.793510, 4.40000, 0.785108, 4.50000, 0.776775, 4.60000,
        0.768509, 4.70000, 0.760310, 4.80000, 0.752179, 4.90000, 0.744114,
        5.00000, 0.736116, 6.00000, 0.659697, 7.00000, 0.589501, 8.00000,
        0.525168, 9.00000, 0.466348, 10.0000, 0.412707, 11.0000, 0.363918,
        12.0000, 0.310828, 13.0000, 0.265483, 14.0000, 0.226753, 15.0000,
        0.193674, 16.0000, 0.165420, 17.0000, 0.141288, 18.0000, 0.120676,
        19.0000, 0.103071, 20.0000, 0.0880349, 21.0000, 0.0748737, 22.0000,
        0.0637273, 23.0000, 0.0542803, 24.0000, 0.0462674, 25.0000, 0.0394658,
        26.0000, 0.0336882, 27.0000, 0.0287769, 28.0000, 0.0245988, 29.0000,
        0.0210420, 30.0000, 0.0180119, 31.0000, 0.0154288, 32.0000, 0.0132250,
        33.0000, 0.0112620, 34.0000, 0.00960889, 35.0000, 0.00821392, 36.0000,
        0.00703441, 37.0000, 0.00603513, 38.0000, 0.00518691, 39.0000,
        0.00446557, 40.0000, 0.00385101, 45.0000, 0.00188129, 50.0000,
        0.000977525, 55.0000, 0.000536684, 60.0000, 0.000288321, 65.0000,
        0.000149342, 70.0000, 0.0000742430, 80.0000, 0.0000157005, 90, 0, 100,
        0, 110, 0
    ])

    alt = Temporary_rho[0::2] * 10**3
    density = Temporary_rho[1::2]

    rho_curr = np.interp(h, alt, density)

    return rho_curr

def model_burn(variables,t,G_c,M_e,R_e,m_dot,thrust_design,rocket_radius,Cd,R_air,R_prop,gamma,P_chamber,T_chamber,alt_design,A_exit,alpha):
    
    v,phi,r,theta,m,delta_vee_step = variables
    
    R_current = r
    altitude = R_current - R_e
    g = (G_c*M_e/R_current**2)
    
    if m_dot == 0:
        T = 0
    else:
        T = thrust_current(altitude,alt_design,m_dot,P_chamber,T_chamber,gamma,R_air,R_prop,g,A_exit)
        
    D = drag_current(v,Cd,altitude,rocket_radius)
    
    v_prime = (T - D)/m - g*np.sin(phi)
    phi_prime = -g*np.cos(phi)/v + v*np.cos(phi)/R_current + T*np.sin(alpha)/(m*v)
    r_prime = v*np.sin(phi)
    theta_prime = v/R_current*np.cos(phi)
    m_prime = -m_dot
    if m_dot == 0:
        delta_vee_prime = 0
    else:
        delta_vee_prime = T/m_dot*np.log(m/(m-m_dot))
        
    return v_prime, phi_prime, r_prime, theta_prime, m_prime, delta_vee_prime
    
def m_dot_design(thrust,alt_current,alt_design,P_chamber,T_chamber,gamma,R_air,R_prop,g_0):

    P_alt,T_alt,rho_alt = current_pressure(alt_current,g_0,R_air)
    P_des,T_des,rho_des = current_pressure(alt_design,g_0,R_air)
    
    AR, M, T = area_ratio_calc(P_chamber,P_alt,gamma,T_chamber)
    
    v_e = exhaust_velocity(gamma,R_prop,T,M)
    Isp = v_e/g_0
    
    rho = P_alt/(T*R_prop)
    
    [m_dot,A_exit] = opt.fsolve(f,(5,0.01),args = (rho,v_e,thrust,P_des,P_alt))
   
    return m_dot, Isp, A_exit

def f(variables,*args):
    (x,y) = variables
    (rho,v_e,thrust,P_des,P_alt) = args
    
    first_eq = x/(rho*v_e) - y
    second_eq = (thrust - y*(P_des-P_alt))/v_e - x
    
    return [first_eq,second_eq]

    
def set_variables(m_dry, stage_mass_ratios):
    
    g = 9.81
    G_c = 6.673e-11
    M_e = 5.98e24
    R_e = 6.38e6
    
    R_star = 8.314459848
    Molar_mass = 28.9645e-3
    R_air = R_star/Molar_mass
    R_prop = 280
    gamma = 1.14
    T_chamber = 3850
    P_chamber = 6e6
    
    t_steps = 1000
    rocket_radius = 0.25
    Cd = 0.2
    target_altitude = 300e3
    
    thrust_design = 1
    alt_design = np.array([15e3,50e3])
    Isp_design = m_dot_design(thrust_design,alt_design[0],alt_design[0],P_chamber,T_chamber,gamma,R_air,R_prop,g)[1]

    orbital_vel = np.sqrt(G_c*M_e/(R_e+target_altitude))
    delta_vee_drag = 200
    delta_vee_gravity = 2000
    
    delta_vee_req = orbital_vel + delta_vee_gravity + delta_vee_drag
    
    stage_m_dry = np.array(m_dry*stage_mass_ratios)
    
    return g, G_c, M_e, R_e, t_steps, Isp_design, thrust_design, rocket_radius, Cd, target_altitude, delta_vee_req, stage_m_dry, orbital_vel, R_air, R_prop, gamma, P_chamber, T_chamber, alt_design

def calculate_trajectory(delta_vee_req, stage_delta_vee_ratios, Isp_design, stage_m_dry, t_steps, g, G_c, M_e, R_e, thrust_design, rocket_radius, Cd, event_alt, GT_angle, R_air, R_prop, gamma, P_chamber, T_chamber, alpha, accel_init, alt_design, coast_on = False):
    
#    Isp_design = np.zeros(2)
#    init_guess_1 = 25e3
#    init_guess_2 = 150e3
    
#    delta_vee_stage = np.array([delta_vee_req]*stage_delta_vee_ratios)
#    print(delta_vee_stage)
    
    for count in range(1):

#        Isp_design[0] = m_dot_design(thrust_design,init_guess_1,P_chamber,T_chamber,gamma,R_air,R_prop,g)[1]
#        Isp_design[1] = m_dot_design(thrust_design,init_guess_2,P_chamber,T_chamber,gamma,R_air,R_prop,g)[1]
        
        
#        mass_ratio = np.exp(delta_vee_stage/(Isp_design*g))
        mass_ratio = stage_delta_vee_ratios
        
        stage_m_prop_2 = (mass_ratio[1]-1)*(stage_m_dry[1])
        stage_m_init_2 = stage_m_prop_2 + stage_m_dry[1]
        stage_m_prop_1 = (mass_ratio[0]-1)*(stage_m_init_2 + (stage_m_dry[0]))
        stage_m_init_1 = stage_m_prop_1 + stage_m_dry[0] + stage_m_init_2
        
        stage_m_init = np.array([stage_m_init_1,stage_m_init_2])
        stage_m_prop = np.array([stage_m_prop_1,stage_m_prop_2])
        
        
        ### First Stage
        thrust_design = (stage_m_init[0]*g)*(1+accel_init[0]) # At sea level
        stage_m_dot = np.zeros(2)
        t_burn = np.zeros(2)
        A_exit = np.zeros(2)
        stage_m_dot[0], Isp, A_exit[0] = m_dot_design(thrust_design,0,alt_design[0],P_chamber,T_chamber,gamma,R_air,R_prop,g)
            
        t_burn[0] = stage_m_prop[0] / stage_m_dot[0]
    
        t1 = np.linspace(0,t_burn[0]/2,t_steps)
        init_conds = [1, np.pi/2, R_e, 0, stage_m_init[0],0]
        trajectory1 = odeint(model_burn, init_conds, t1, args=(G_c,M_e,R_e,stage_m_dot[0],thrust_design,rocket_radius,Cd,R_air,R_prop,gamma,P_chamber,T_chamber,alt_design[0],A_exit[0],0))
        
        [v,phi,r,theta,m,delta_vee_step] = np.transpose(trajectory1)
        
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        
        ind = np.where(np.abs(r-event_alt-R_e) == np.min(np.abs(r-event_alt-R_e)))[0][0]
        
        try:
            ind = np.where(np.abs(r-event_alt-R_e) == np.min(np.abs(r-event_alt-R_e)))[0][0]
        except IndexError as error:
            ind = 1
            
        if ind == 0:
            ind = 1
        
        event_conds = [v[ind],GT_angle,r[ind],theta[ind],m[ind],delta_vee_step[ind]]
        t_event = t1[ind]
        t2 = np.linspace(t_event,t_burn[0],t_steps)
        
        trajectory2 = odeint(model_burn, event_conds, t2, args=(G_c,M_e,R_e,stage_m_dot[0],thrust_design,rocket_radius,Cd,R_air, R_prop,gamma,P_chamber,T_chamber,alt_design[0],A_exit[0],alpha))
        
        [v,phi,r,theta,m,delta_vee_step] = np.concatenate((np.transpose(trajectory1[:ind,:]),np.transpose(trajectory2)),axis = 1)
        
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        
#        delta_vee_1 = delta_vee_step[-1]
        
        
        
        
        ### Second Stage
        thrust_design = (stage_m_init[1]*g*np.sin(phi[-1]))*(1+accel_init[1])
        
        altitude = r[-1] - R_e
        
        stage_m_dot[1], Isp, A_exit[1] = m_dot_design(thrust_design,altitude,alt_design[1],P_chamber,T_chamber,gamma,R_air,R_prop,g)
            
        t_burn[1] = stage_m_prop[1] / stage_m_dot[1]
        
        second_stage_conds = [v[-1],phi[-1],r[-1],theta[-1],stage_m_init[1],0]
        t3 = np.linspace(t_burn[0],t_burn[0] + t_burn[1],t_steps)
        
        trajectory3 = odeint(model_burn, second_stage_conds, t3, args=(G_c,M_e,R_e,stage_m_dot[1],thrust_design,rocket_radius,Cd,R_air, R_prop,gamma,P_chamber,T_chamber,alt_design[0],A_exit[1],alpha))
        
        [v,phi,r,theta,m,delta_vee_step] = np.concatenate((np.transpose(trajectory1[:ind-1,:]),np.transpose(trajectory2[:-1,:]),np.transpose(trajectory3[:-1,:])),axis = 1)
        
        t = np.concatenate((t1[:ind-1],t2[:-1],t3[:-1]))
        
#        delta_vee_2 = delta_vee_step[-1]
#        print('alt_guess',init_guess_1,init_guess_2)
#        print('resultant dv',delta_vee_1,delta_vee_2)
    
#        init_guess_1 = init_guess_1*(1 - 5*(delta_vee_stage[0]-delta_vee_1)/delta_vee_stage[0])
#        init_guess_2 = init_guess_2*(1 - 5*(delta_vee_stage[1]-delta_vee_2)/delta_vee_stage[1])
#        print(t[-1])
    
    ### Coast
    if coast_on:
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        
        coast_conds = [v[-1],phi[-1],r[-1],theta[-1],m[-1],10]
#        print(coast_conds)
        t4 = np.linspace(t_burn[0] + t_burn[1],(t_burn[0] + t_burn[1])*21,t_steps*20)
#        print(t4)

        trajectory4 = odeint(model_burn, coast_conds, t4, args=(G_c,M_e,R_e,0,0,rocket_radius,Cd,R_air, R_prop,gamma,P_chamber,T_chamber,alt_design[0],A_exit[1],0))
        
        
        [v,phi,r,theta,m,delta_vee_step] = np.concatenate((np.transpose(trajectory1[:ind-1,:]),np.transpose(trajectory2[:-1]),np.transpose(trajectory3[:-1]),np.transpose(trajectory4)),axis = 1)
        
        t = np.concatenate((t1[:ind-1],t2[:-1],t3[:-1],t4))
    
    return v, phi, r, theta, m, t, ind

def trajectory_score(v,phi,r,m,R_e,target_altitude,orbital_vel,weights):
    
    v_final = v[-1]
    alt_final = r[-1]-R_e
    angle_final = phi[-1]
    m_initial = m[0]
    
    target_angle = 0
    vel_adj = 10
    alt_adj = 1e3
    ang_adj = 1*np.pi/180
    mass_adj = 100
    
    vel_factor = abs(v_final - orbital_vel)/(vel_adj)
    alt_factor = abs(alt_final - target_altitude)/(alt_adj)
    angle_factor = abs(angle_final - target_angle)/(ang_adj)
    
    mass_score = abs(m_initial)/mass_adj
    
    score = (vel_factor*weights[0]) + (alt_factor*weights[1]) +(angle_factor*weights[2]) + (mass_score*weights[3])
    
    print(score)      
    
        
    return score, [v_final, alt_final, angle_final, m_initial]

def run_trajectory(inputs,m_dry,stage_mass_ratios,GT_angle,weights):
    
    [stage_delta_vee_ratios_1,stage_delta_vee_ratios_2,event_alt,accel_init_1,accel_init_2,alpha] = inputs
    stage_delta_vee_ratios = np.array([stage_delta_vee_ratios_1,stage_delta_vee_ratios_2])
    accel_init = np.array([accel_init_1,accel_init_2])
    g, G_c, M_e, R_e, t_steps, Isp_design, thrust_design, rocket_radius, Cd, target_altitude, delta_vee_req, stage_m_dry, orbital_vel, R_air, R_prop, gamma, P_chamber, T_chamber, alt_design = set_variables(m_dry, stage_mass_ratios)
    v, phi, r, theta, m, t, ind = calculate_trajectory(delta_vee_req, stage_delta_vee_ratios, Isp_design, stage_m_dry, t_steps, g, G_c, M_e, R_e, thrust_design, rocket_radius, Cd, event_alt, GT_angle, R_air, R_prop, gamma, P_chamber, T_chamber, alpha, accel_init, alt_design)
   
    score, results = trajectory_score(v,phi,r,m,R_e,target_altitude,orbital_vel,weights)
    
    return score

def run_trajectory_final(m_dry,stage_mass_ratios,event_alt,GT_angle,stage_delta_vee_ratios,weights,alpha,accel_init):
    
    g, G_c, M_e, R_e, t_steps, Isp_design, thrust_design, rocket_radius, Cd, target_altitude, delta_vee_req, stage_m_dry, orbital_vel, R_air, R_prop, gamma, P_chamber, T_chamber, alt_design = set_variables(m_dry, stage_mass_ratios)
    v, phi, r, theta, m, t, ind = calculate_trajectory(delta_vee_req, stage_delta_vee_ratios, Isp_design, stage_m_dry, t_steps, g, G_c, M_e, R_e, thrust_design, rocket_radius, Cd, event_alt, GT_angle, R_air, R_prop, gamma, P_chamber, T_chamber, alpha, accel_init, alt_design, True)

    score = trajectory_score(v,phi,r,m,R_e,target_altitude,orbital_vel,weights)
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    grav_delta_vee = np.sum(g*np.sin(phi[:-t_steps*20])*(t[1:-t_steps*20+1] - t[:-t_steps*20]))
    
    
    plt.figure()
    plt.plot(x/1000,(y)/1000)
    plt.axis('equal')
    plt.xlabel('Downrange (km)')
    plt.ylabel('Altitude (km)')
    
    altitudes = r-R_e
    
    p1 = ind
    p2 = ind + t_steps - 2
    p3 = ind + 2*t_steps - 4
    
    plt.figure()
    plt.plot(t,altitudes/1000)
    plt.scatter(np.array([t[p1],t[p2],t[p3]]),np.array([altitudes[p1],altitudes[p2],altitudes[p3]])/1000,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    
    plt.figure()
    plt.plot(t,phi*180/np.pi)
    plt.scatter(np.array([t[p1],t[p2],t[p3]]),np.array([phi[p1],phi[p2],phi[p3]])*180/np.pi,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Flight Angle (deg)')
    
    plt.figure()
    plt.plot(t,theta*180/np.pi)
    plt.xlabel('Time (s)')
    plt.ylabel('Central Angle (deg)')
    
    plt.figure()
    plt.plot(t,v/1000)
    plt.scatter(np.array([t[p1],t[p2],t[p3]]),np.array([v[p1],v[p2],v[p3]])/1000,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (km/s)')
    
    
    
#    print('Initial Acceleration = {}g,{}g'.format((stage_m_dot[0]*Isp_design*g/(g*stage_m_init[0])-1),(stage_m_dot[1]*Isp_design*g/(g*stage_m_init[1])-1)))
    print('Initial Tilt Angle = {} degrees'.format(phi[p1]*180/np.pi))
    print('Burn End Angle = {} degrees'.format(phi[p3]*180/np.pi))
    print('Burn End Position Tangent = {} degrees'.format((np.arctan2(y,x)[p3]-np.pi/2)*180/np.pi))
    print('Burn End Altitude = {}km'.format((np.sqrt(x**2 + y**2)-R_e)[p3]/1000))
    print('Burn End Velocity = {}m/s'.format(v[p3]))
    print('Initial Mass = {}kg'.format(m[0]))
    
    return v, phi, r, theta, m, t, ind, grav_delta_vee, score

if __name__ == "__main__":
    
    import time
    import ML_GT_Analysis as GTA
    start_t = time.time()
#    plt.close('all')
    coast_on = False
    ## 
    
    filename = 'Population_Results.npy'
    pop_ref_all = np.load(filename).item()
    pop_best = pop_ref_all['actions'][0]
            
    input_variables = pop_best

    m_dry, event_alt, GT_angle, stage_mass_ratios, stage_delta_vee_ratios, alpha, accel_init = GTA.get_input_variables(input_variables, GT_angle_max)
    
    weights = [1.0,1.0,1.0]
    
    g, G_c, M_e, R_e, t_steps, Isp_design, thrust_design, rocket_radius, Cd, target_altitude, delta_vee_req, stage_m_dry, stage_m_dot, orbital_vel, R_air, R_prop, gamma, P_chamber, T_chamber = set_variables(m_dry, stage_mass_ratios, stage_m_dots)
    v, phi, r, theta, m, t, ind = calculate_trajectory(delta_vee_req, stage_delta_vee_ratios, Isp_design, stage_m_dry, stage_m_dot, t_steps, g, G_c, M_e, R_e, thrust_design, rocket_radius, Cd, event_alt, GT_angle,R_air, R_prop,gamma,P_chamber,T_chamber,alpha, coast_on)

    score = trajectory_score(v,phi,r,m,R_e,target_altitude,orbital_vel,weights)
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    grav_delta_vee = np.sum(g*np.cos(phi[:-t_steps])*(t[1:] - t[:-1]))
    
    total_time = time.time() - start_t 
    print(total_time)
    
    plt.figure()
    plt.plot(x/1000,(y)/1000)
    plt.axis('equal')
    plt.xlabel('Downrange (km)')
    plt.ylabel('Altitude (km)')
    
    altitudes = r-R_e
    
    p1 = ind
    p2 = ind + t_steps - 2
    p3 = ind + 2*t_steps - 4
    
    plt.figure()
    plt.plot(t,altitudes/1000)
    plt.scatter(np.array([t[p1],t[p2],t[p3]]),np.array([altitudes[p1],altitudes[p2],altitudes[p3]])/1000,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    
    plt.figure()
    plt.plot(t,phi*180/np.pi)
    plt.scatter(np.array([t[p1],t[p2],t[p3]]),np.array([phi[p1],phi[p2],phi[p3]])*180/np.pi,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Flight Angle (deg)')
    
    plt.figure()
    plt.plot(t,theta*180/np.pi)
    plt.xlabel('Time (s)')
    plt.ylabel('Central Angle (deg)')
    
    plt.figure()
    plt.plot(t,v/1000)
    plt.scatter(np.array([t[p1],t[p2],t[p3]]),np.array([v[p1],v[p2],v[p3]])/1000,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (km/s)')
    
    
    
#    print('Initial Acceleration = {}g,{}g'.format((stage_m_dot[0]*Isp_design*g/(g*stage_m_init[0])-1),(stage_m_dot[1]*Isp_design*g/(g*stage_m_init[1])-1)))
    print('Initial Tilt Angle = {} degrees'.format(phi[p1]*180/np.pi))
    print('Burn End Angle = {} degrees'.format(phi[p3]*180/np.pi))
    print('Burn End Position Tangent = {} degrees'.format((np.arctan2(y,x)[p3]-np.pi/2)*180/np.pi))
    print('Burn End Altitude = {}km'.format((np.sqrt(x**2 + y**2)-R_e)[p3]/1000))
    print('Burn End Velocity = {}m/s'.format(v[p3]))
    

