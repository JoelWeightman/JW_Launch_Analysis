# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:57:53 2018

@author: JL
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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

def current_pressure(altitude):
    
    Temporary_pressure = np.array([0.00000000e+00, 1.01325000e+05, 1.00000000e-01, 1.00129426e+05,
       2.00000000e-01, 9.89453020e+04, 3.00000000e-01, 9.77725386e+04,
       4.00000000e-01, 9.66110579e+04, 5.00000000e-01, 9.54607764e+04,
       6.00000000e-01, 9.43216059e+04, 7.00000000e-01, 9.31934679e+04,
       8.00000000e-01, 9.20762840e+04, 9.00000000e-01, 9.09699708e+04,
       1.00000000e+00, 8.98744449e+04, 1.10000000e+00, 8.87896231e+04,
       1.20000000e+00, 8.77154267e+04, 1.30000000e+00, 8.66517824e+04,
       1.40000000e+00, 8.55986067e+04, 1.50000000e+00, 8.45558163e+04,
       1.60000000e+00, 8.35233377e+04, 1.70000000e+00, 8.25010923e+04,
       1.80000000e+00, 8.14890017e+04, 1.90000000e+00, 8.04869875e+04,
       2.00000000e+00, 7.94949713e+04, 2.10000000e+00, 7.85128793e+04,
       2.20000000e+00, 7.75406348e+04, 2.30000000e+00, 7.65781622e+04,
       2.40000000e+00, 7.56253859e+04, 2.50000000e+00, 7.46822310e+04,
       2.60000000e+00, 7.37486228e+04, 2.70000000e+00, 7.28244875e+04,
       2.80000000e+00, 7.19097513e+04, 2.90000000e+00, 7.10043413e+04,
       3.00000000e+00, 7.01081845e+04, 3.10000000e+00, 6.92212076e+04,
       3.20000000e+00, 6.83433392e+04, 3.30000000e+00, 6.74745077e+04,
       3.40000000e+00, 6.66146415e+04, 3.50000000e+00, 6.57636694e+04,
       3.60000000e+00, 6.49215209e+04, 3.70000000e+00, 6.40881258e+04,
       3.80000000e+00, 6.32634146e+04, 3.90000000e+00, 6.24473181e+04,
       4.00000000e+00, 6.16397671e+04, 4.10000000e+00, 6.08406930e+04,
       4.20000000e+00, 6.00500272e+04, 4.30000000e+00, 5.92677019e+04,
       4.40000000e+00, 5.84936501e+04, 4.50000000e+00, 5.77278040e+04,
       4.60000000e+00, 5.69700970e+04, 4.70000000e+00, 5.62204633e+04,
       4.80000000e+00, 5.54788368e+04, 4.90000000e+00, 5.47451518e+04,
       5.00000000e+00, 5.40193430e+04, 6.00000000e+00, 4.71751807e+04,
       7.00000000e+00, 4.10499225e+04, 8.00000000e+00, 3.55843066e+04,
       9.00000000e+00, 3.07225526e+04, 1.00000000e+01, 2.64122339e+04,
       1.10000000e+01, 2.26041652e+04, 1.20000000e+01, 1.92956403e+04,
       1.30000000e+01, 1.64697779e+04, 1.40000000e+01, 1.40561658e+04,
       1.50000000e+01, 1.19946594e+04, 1.60000000e+01, 1.02338909e+04,
       1.70000000e+01, 8.72999010e+03, 1.80000000e+01, 7.44548536e+03,
       1.90000000e+01, 6.34837247e+03, 2.00000000e+01, 5.41131076e+03,
       2.10000000e+01, 4.61251049e+03, 2.20000000e+01, 3.93289928e+03,
       2.30000000e+01, 3.35426503e+03, 2.40000000e+01, 2.86124305e+03,
       2.50000000e+01, 2.44086197e+03, 2.60000000e+01, 2.08216129e+03,
       2.70000000e+01, 1.77587220e+03, 2.80000000e+01, 1.51415175e+03,
       2.90000000e+01, 1.29035833e+03, 3.00000000e+01, 1.09886287e+03,
       3.10000000e+01, 9.34890987e+02, 3.20000000e+01, 7.94391013e+02,
       3.30000000e+01, 6.74322330e+02, 3.40000000e+01, 5.71984763e+02,
       3.50000000e+01, 4.84593049e+02, 3.60000000e+01, 4.09824952e+02,
       3.70000000e+01, 3.45740260e+02, 3.80000000e+01, 2.90714534e+02,
       3.90000000e+01, 2.43384934e+02, 4.00000000e+01, 2.02605738e+02,
       4.50000000e+01, 6.20681221e+01, 5.00000000e+01, 0.00000000e+00,
       5.50000000e+01, 0.00000000e+00, 6.00000000e+01, 0.00000000e+00,
       6.50000000e+01, 0.00000000e+00, 7.00000000e+01, 0.00000000e+00,
       8.00000000e+01, 0.00000000e+00, 9.00000000e+01, 0.00000000e+00,
       1.00000000e+02, 0.00000000e+00, 1.10000000e+02, 0.00000000e+00])
    
    alt = Temporary_pressure[0::2] * 10**3
    pressure = Temporary_pressure[1::2]

    Pressure_current = np.interp(altitude, alt, pressure)
    
    return Pressure_current

def thrust_current(alt,Isp_design,m_dot,P_chamber,T_chamber,gamma,R):
    
    P_alt = current_pressure(alt)
    
    AR, M, T = area_ratio_calc(P_chamber,P_alt,gamma,T_chamber)
    
    v_e = exhaust_velocity(gamma,R,T,M)
    
    thrust = m_dot*v_e
    
    return thrust,m_dot

def drag_current(V, Cd, d, rocket_diameter):

    Area = rocket_diameter**2*np.pi
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

def model_burn(variables,t,G_c,M_e,R_e,m_dot,thrust_design,rocket_diam,Cd,R,gamma,P_chamber,T_chamber):
    
    v,phi,r,theta,m = variables
    
    R_current = r
    altitude = R_current - R_e
    T = thrust_current(altitude,Isp_design,m_dot,P_chamber,T_chamber,gamma,R)
    D = drag_current(v,Cd,altitude,rocket_diam)
    
    v_prime = (T - D)/m - (G_c*M_e/R_current**2)*np.sin(phi)
    phi_prime = -1/v*(G_c*M_e/R_current**2)*np.cos(phi) + v*np.cos(phi)/R_current
    r_prime = v*np.sin(phi)
    theta_prime = v/R_current*np.cos(phi)
    m_prime = -m_dot
    
    return v_prime, phi_prime, r_prime, theta_prime, m_prime
    
    
def set_variables(m_dry, stage_mass_ratios, stage_m_dots):
    
    g = 9.81
    G_c = 6.673e-11
    M_e = 5.98e24
    R_e = 6.38e6
    
    R = 287
    gamma = 1.4
    T_chamber = 3850
    P_chamber = 6e6
    
    t_steps = 1000
    Isp_design = 330
    thrust_design = 50000
    rocket_diam = 0.5
    Cd = 0.2
    target_altitude = 300e3
    
    orbital_vel = np.sqrt(G_c*M_e/(R_e+target_altitude))
    delta_vee_drag = 250
    delta_vee_gravity = 2500
    
    delta_vee_req = orbital_vel + delta_vee_gravity + delta_vee_drag
    
    stage_m_dry = np.array(m_dry*stage_mass_ratios)
    stage_m_dot = np.array(stage_m_dots)*thrust_design/(Isp_design*g)

    return g, G_c, M_e, R_e, t_steps, Isp_design, thrust_design, rocket_diam, Cd, target_altitude, delta_vee_req, stage_m_dry, stage_m_dot, orbital_vel, R, gamma, P_chamber, T_chamber

def calculate_trajectory(delta_vee_req, stage_delta_vee_ratios, Isp_design, stage_m_dry, stage_m_dot, t_steps, g, G_c, M_e, R_e, thrust_design, rocket_diam, Cd, event_alt, GT_angle, R, gamma, P_chamber, T_chamber, coast_on = False):
    
    delta_vee_stage = np.array([delta_vee_req]*stage_delta_vee_ratios)
    mass_ratio = np.exp(delta_vee_stage/(Isp_design*g))/1.2
    
    stage_m_prop_2 = (mass_ratio[1]-1)*(stage_m_dry[1])
    stage_m_init_2 = stage_m_prop_2 + stage_m_dry[1]
    stage_m_prop_1 = (mass_ratio[0]-1)*(stage_m_init_2 + (stage_m_dry[0]-stage_m_dry[1]))
    stage_m_init_1 = stage_m_prop_1 + stage_m_dry[0] - stage_m_dry[1] + stage_m_init_2
    
    stage_m_init = np.array([stage_m_init_1,stage_m_init_2])
    stage_m_prop = np.array([stage_m_prop_1,stage_m_prop_2])
     
    t_burn = stage_m_prop / stage_m_dot
    
    t1 = np.linspace(0,t_burn[0]/2,t_steps)
    init_conds = [1, np.pi/2, R_e, 0, stage_m_init[0]]
    trajectory1 = odeint(model_burn, init_conds, t1, args=(G_c,M_e,R_e,stage_m_dot[0],thrust_design,rocket_diam,Cd,R,gamma,P_chamber,T_chamber))
    
    [v,phi,r,theta,m] = np.transpose(trajectory1)
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    ind = np.where(np.abs(r-event_alt-R_e) == np.min(np.abs(r-event_alt-R_e)))[0][0]
    
    if ind == 0:
        ind = 1
    
    event_conds = [v[ind],GT_angle,r[ind],theta[ind],m[ind]]
    t_event = t1[ind]
    t2 = np.linspace(t_event,t_burn[0],t_steps)
    
    trajectory2 = odeint(model_burn, event_conds, t2, args=(G_c,M_e,R_e,stage_m_dot[0],thrust_design,rocket_diam,Cd,R,gamma,P_chamber,T_chamber))
    
    [v,phi,r,theta,m] = np.concatenate((np.transpose(trajectory1[:ind,:]),np.transpose(trajectory2)),axis = 1)
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    second_stage_conds = [v[-1],phi[-1],r[-1],theta[-1],stage_m_init[1]]
    t3 = np.linspace(t_burn[0],t_burn[0] + t_burn[1],t_steps)
    
    trajectory3 = odeint(model_burn, second_stage_conds, t3, args=(G_c,M_e,R_e,stage_m_dot[1],stage_m_dot[1]*Isp_design*g,rocket_diam,Cd,R,gamma,P_chamber,T_chamber))
    
    [v,phi,r,theta,m] = np.concatenate((np.transpose(trajectory1[:ind-1,:]),np.transpose(trajectory2[:-1,:]),np.transpose(trajectory3[:-1,:])),axis = 1)
    
    t = np.concatenate((t1[:ind-1],t2[:-1],t3[:-1]))
    
    if coast_on:
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        
        coast_conds = [v[-1],phi[-1],r[-1],theta[-1],m[-1]]
        t4 = np.linspace(t_burn[0] + t_burn[1],(t_burn[0] + t_burn[1])*20,t_steps)
        
        trajectory4 = odeint(model_burn, coast_conds, t4, args=(G_c,M_e,R_e,0,0,rocket_diam,Cd,R,gamma,P_chamber,T_chamber))
        
        [v,phi,r,theta,m] = np.concatenate((np.transpose(trajectory1[:ind-1,:]),np.transpose(trajectory2[:-1]),np.transpose(trajectory3[:-1]),np.transpose(trajectory4)),axis = 1)
        
        t = np.concatenate((t1[:ind-1],t2[:-1],t3[:-1],t4))
    
    return v, phi, r, theta, m, t, ind

def trajectory_score(v,phi,r,R_e,target_altitude,orbital_vel,weights):
    
    v_final = v[-1]
    alt_final = r[-1]-R_e
    angle_final = phi[-1]
    
    target_angle = 0
    vel_adj = 5
    alt_adj = 10
    ang_adj = 10
    
    vel_factor = abs(v_final - orbital_vel)/(orbital_vel/vel_adj)
    alt_factor = abs(alt_final - target_altitude)/(target_altitude/alt_adj)
    angle_factor = abs(angle_final - target_angle)/(np.pi/ang_adj)
    
    score = (weights[0] - vel_factor*weights[0]) + (weights[1] - alt_factor*weights[1]) +(weights[2] - angle_factor*weights[2])
    
    return score, [v_final, alt_final, angle_final]

def run_trajectory(m_dry,stage_mass_ratios,stage_m_dots,event_alt,GT_angle,stage_delta_vee_ratios,weights):
    
    g, G_c, M_e, R_e, t_steps, Isp_design, thrust_design, rocket_diam, Cd, target_altitude, delta_vee_req, stage_m_dry, stage_m_dot, orbital_vel = set_variables(m_dry, stage_mass_ratios, stage_m_dots)
    v, phi, r, theta, m, t, ind = calculate_trajectory(delta_vee_req, stage_delta_vee_ratios, Isp_design, stage_m_dry, stage_m_dot, t_steps, g, G_c, M_e, R_e, thrust_design, rocket_diam, Cd, event_alt, GT_angle)

    score = trajectory_score(v,phi,r,R_e,target_altitude,orbital_vel,weights)
    
    return score

def run_trajectory_final(m_dry,stage_mass_ratios,stage_m_dots,event_alt,GT_angle,stage_delta_vee_ratios,weights):
    
    g, G_c, M_e, R_e, t_steps, Isp_design, thrust_design, rocket_diam, Cd, target_altitude, delta_vee_req, stage_m_dry, stage_m_dot, orbital_vel, R, gamma, P_chamber, T_chamber = set_variables(m_dry, stage_mass_ratios, stage_m_dots)
    v, phi, r, theta, m, t, ind = calculate_trajectory(delta_vee_req, stage_delta_vee_ratios, Isp_design, stage_m_dry, stage_m_dot, t_steps, g, G_c, M_e, R_e, thrust_design, rocket_diam, Cd, event_alt, GT_angle, True)

    score = trajectory_score(v,phi,r,R_e,target_altitude,orbital_vel,weights)
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    grav_delta_vee = np.sum(g*np.cos(phi[:-1])*(t[1:] - t[:-1]))
    
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
    plt.scatter(np.array([t[p1],t[p2]]),np.array([altitudes[p1],altitudes[p2]])/1000,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    
    plt.figure()
    plt.plot(t,phi*180/np.pi)
    plt.scatter(np.array([t[p1],t[p2]]),np.array([phi[p1],phi[p2]])*180/np.pi,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Flight Angle (deg)')
    
    plt.figure()
    plt.plot(t,theta*180/np.pi)
    plt.xlabel('Time (s)')
    plt.ylabel('Central Angle (deg)')
    
    plt.figure()
    plt.plot(t,v/1000)
    plt.scatter(np.array([t[p1],t[p2]]),np.array([v[p1],v[p2]])/1000,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (km/s)')
    
    
    
#    print('Initial Acceleration = {}g,{}g'.format((stage_m_dot[0]*Isp_design*g/(g*stage_m_init[0])-1),(stage_m_dot[1]*Isp_design*g/(g*stage_m_init[1])-1)))
    print('Initial Tilt Angle = {} degrees'.format(phi[p1]*180/np.pi))
    print('Burn End Angle = {} degrees'.format(phi[p3]*180/np.pi))
    print('Burn End Position Tangent = {} degrees'.format((np.arctan2(y,x)[p3]-np.pi/2)*180/np.pi))
    print('Burn End Altitude = {}km'.format((np.sqrt(x**2 + y**2)-R_e)[p3]/1000))
    print('Burn End Velocity = {}m/s'.format(v[p3]))
    
    return v, phi, r, theta, m, t, ind, grav_delta_vee, score

if __name__ == "__main__":
    
    plt.close('all')
        
    ## 
    m_dry,stage_mass_ratios,stage_m_dots,event_alt,GT_angle,stage_delta_vee_ratios1,stage_delta_vee_ratios2 = [0.3023023,  0.84006874, 0.07477426, 0.66729451, 0.70071262, 0.64289019, 0.64272017]
    
    m_dry_max = 0.5e3
    event_alt_max = 5e3
    
    m_dry *= m_dry_max
    event_alt *= event_alt_max
    GT_angle *= np.pi/2
    stage_mass_ratios = np.array([1.0,stage_mass_ratios])
    stage_m_dots = np.array([1.0,stage_m_dots])
    stage_delta_vee_ratios = np.array([stage_delta_vee_ratios1,stage_delta_vee_ratios2])/(stage_delta_vee_ratios1 + stage_delta_vee_ratios2)
        

    weights = [1.0,1.0,1.0]
    
    g, G_c, M_e, R_e, t_steps, Isp_design, thrust_design, rocket_diam, Cd, target_altitude, delta_vee_req, stage_m_dry, stage_m_dot, orbital_vel, R, gamma, P_chamber, T_chamber = set_variables(m_dry, stage_mass_ratios, stage_m_dots)
    v, phi, r, theta, m, t, ind = calculate_trajectory(delta_vee_req, stage_delta_vee_ratios, Isp_design, stage_m_dry, stage_m_dot, t_steps, g, G_c, M_e, R_e, thrust_design, rocket_diam, Cd, event_alt, GT_angle,R,gamma,P_chamber,T_chamber, False)

    score = trajectory_score(v,phi,r,R_e,target_altitude,orbital_vel,weights)
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    grav_delta_vee = np.sum(g*np.cos(phi[:-1])*(t[1:] - t[:-1]))
    
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
    plt.scatter(np.array([t[p1],t[p2]]),np.array([altitudes[p1],altitudes[p2]])/1000,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    
    plt.figure()
    plt.plot(t,phi*180/np.pi)
    plt.scatter(np.array([t[p1],t[p2]]),np.array([phi[p1],phi[p2]])*180/np.pi,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Flight Angle (deg)')
    
    plt.figure()
    plt.plot(t,theta*180/np.pi)
    plt.xlabel('Time (s)')
    plt.ylabel('Central Angle (deg)')
    
    plt.figure()
    plt.plot(t,v/1000)
    plt.scatter(np.array([t[p1],t[p2]]),np.array([v[p1],v[p2]])/1000,10,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (km/s)')
    
    
    
#    print('Initial Acceleration = {}g,{}g'.format((stage_m_dot[0]*Isp_design*g/(g*stage_m_init[0])-1),(stage_m_dot[1]*Isp_design*g/(g*stage_m_init[1])-1)))
    print('Final Angle = {}'.format(phi[p2]*180/np.pi))
    print('Final Position Tangent = {}'.format(np.arctan2(y,x)[-1]-np.pi/2))
    print('Final Altitude = {}'.format((np.sqrt(x**2 + y**2)-R_e)[-1]))
    print('Final Velocity = {}'.format(v[-1]))
    
