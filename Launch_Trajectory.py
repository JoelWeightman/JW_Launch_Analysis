# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:47:23 2018

@author: JL
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:39:09 2018

@author: joellukw
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:44:07 2017

@author: JL

"""

# Ballistic Code for SpaceOps
import numpy as np
import matplotlib.pyplot as plt


def drag_current(V, Cd, d, rocket_diameter):

    Area = rocket_diameter**2*np.pi
    rho_curr = density_calc(d)
    drag = 0.5 * rho_curr * V * V * Cd * Area
#    print(V)
    return drag, rho_curr


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

def SOS_calc(h):
    Temporary_SOS = np.array([0, 340.294, 0.5, 338.37, 1,	336.434, 1.5,	334.487,
2,	332.529, 2.5,	330.56, 3,	328.578,
3.5,	326.584, 4,	324.579, 4.5,	322.56, 5,	320.529, 5.5,	318.486, 6,	316.428,
6.5,	314.358, 7,	312.274, 7.5,	310.175, 8,	308.063, 8.5,	305.935, 9,	303.793,
9.5,	301.636, 10,	299.463, 10.5, 297.275, 11, 295.07, 20,	295.07,
25,	298.455, 30,	301.803, 35,	308.649, 40,	317.633, 45,	326.369, 50,	329.8,
55,	322.903, 60,	314.07,65, 304.982, 70,	295.614,75,	288.179,80,	281.12,
80.5,	280.405,85,	274.096,85.5,	274.096,86,	274.096])
    alt = Temporary_SOS[0::2] * 10**3
    
    SOS = Temporary_SOS[1::2]

    SOS_curr = np.interp(h, alt, SOS)
    
    return SOS_curr

def initial_values(m_dot, Isp, g, m_init, d_init, t_steps, Cd, rocket_diameter,t):
    thrust = m_dot*Isp*g
    
    m = np.zeros(np.shape(t))
    d_x = np.zeros(np.shape(t))
    v_x = np.zeros(np.shape(t))
    a_x = np.zeros(np.shape(t))
    d_y = np.zeros(np.shape(t))
    v_y = np.zeros(np.shape(t))
    a_y = np.zeros(np.shape(t))
    drag = np.zeros(np.shape(t))
    rho = np.zeros(np.shape(t))

    m[0] = m_init
    d_x[0] = 0
    v_x[0] = 0
    a_x[0] = 0
    d_y[0] = d_init
    v_y[0] = 0
    a_y[0] = (thrust - (m[0]*g))/m[0]
    v = np.sqrt(v_y[0]**2 + v_x[0]**2)
    drag[0], rho[0] = drag_current(v, Cd, d_y[0], rocket_diameter)

    return m, d_x, d_y, v_x, v_y, a_x, a_y, drag, rho


def launch_traj(t, m, v_x, v_y, d_x, d_y, drag, rho, a_x, a_y, g, Cd, rocket_diameter, turn_point, R_e, coast_to_burn, stage, stage_m_dry, stage_m_init, stage_m_prop, stage_m_dot, stage_h, Isp, thrust_profile, delta_vee_g_on_t):

    dt = (t[1] - t[0])
    v = np.zeros(np.shape(v_x))
    Mach = np.zeros(np.shape(v_x))
    theta_gravity = np.zeros(np.shape(v_x))
    theta = np.zeros(np.shape(v_x))
    
    initial_angle_delta = 0.5
    initial_angle = 15
    theta = (np.linspace(90-initial_angle_delta,90-initial_angle_delta,np.shape(t)[0])*np.pi/180)
    
    count = 0
    
    m_dot = stage_m_dot[stage-1]
    
    thrust = m_dot*Isp*g*thrust_profile
    i = np.where(m == np.min(m[m!= 0]))[0][0]+1
    
    if stage < np.size(stage_m_init):
        target_mass = stage_m_init[stage] + (stage_m_dry[stage-1]-stage_m_dry[stage])
    else:
        target_mass = stage_m_dry[-1]

    if stage > 1:
        m[i-1] = stage_m_init[stage-1]

    for i in range(i,np.size(t)):
        if d_y[i-1] < turn_point:
            v_x[i] = v_x[i - 1] + a_x[i - 1] * dt
            d_x[i] = d_x[i - 1] + v_x[i - 1] * dt
            v_y[i] = v_y[i - 1] + a_y[i - 1] * dt
            d_y[i] = d_y[i - 1] + v_y[i - 1] * dt
            m[i] = m[i - 1] - m_dot * dt
            v[i] = np.sqrt(v_y[i]**2 + v_x[i]**2)
            drag[i], rho[i] = drag_current(v[i], Cd, d_y[i], rocket_diameter)
            theta_gravity[i] = np.arctan2(d_x[i],R_e + d_y[i])
            a_x[i] = (thrust[i] * np.cos(theta[i]) - m[i] * g * np.sin(theta_gravity[i]) - drag[i] * np.cos(theta[i])) / m[i]
            a_y[i] = (thrust[i] * np.sin(theta[i]) - m[i] * g * np.cos(theta_gravity[i]) - drag[i] * np.sin(theta[i])) / m[i]
            SOS = SOS_calc(d_y[i])
            Mach[i] = v[i]/SOS
            delta_vee_g_on_t[i] = g * np.cos(theta_gravity[i])
        else:
            if count == 0:
                theta = np.concatenate((np.linspace(90-initial_angle_delta,90-initial_angle_delta,i-1)*np.pi/180,np.linspace(90-initial_angle,90-initial_angle,np.shape(t)[0]-i+1)*np.pi/180),axis = 0)
                count += 1
            v_x[i] = v_x[i - 1] + a_x[i - 1] * dt
            d_x[i] = d_x[i - 1] + v_x[i - 1] * dt
            v_y[i] = v_y[i - 1] + a_y[i - 1] * dt
            d_y[i] = d_y[i - 1] + v_y[i - 1] * dt
            m[i] = m[i - 1] - m_dot * dt
            v[i] = np.sqrt(v_y[i]**2 + v_x[i]**2)
            drag[i], rho[i] = drag_current(v[i], Cd, d_y[i], rocket_diameter)
            theta_gravity[i] = np.arctan2(d_x[i],R_e + d_y[i])
            if theta[i] > np.arctan2(v_y[i],v_x[i]):
                theta[i] = np.arctan2(v_y[i],v_x[i])
            a_x[i] = (thrust[i] * np.cos(theta[i]) - m[i] * g * np.sin(theta_gravity[i]) - drag[i] * np.cos(theta[i])) / m[i]
            a_y[i] = (thrust[i] * np.sin(theta[i]) - m[i] * g * np.cos(theta_gravity[i]) - drag[i] * np.sin(theta[i])) / m[i]
            SOS = SOS_calc(d_y[i])
            Mach[i] = v[i]/SOS
            delta_vee_g_on_t[i] = g * np.cos(theta_gravity[i])
        if m[i] < target_mass:
            break
                
    if m[i] <= stage_m_dry[-1]:
        
        for i in range(i,np.size(t)):
            v_x[i] = v_x[i - 1] + a_x[i - 1] * dt
            d_x[i] = d_x[i - 1] + v_x[i - 1] * dt
            v_y[i] = v_y[i - 1] + a_y[i - 1] * dt
            d_y[i] = d_y[i - 1] + v_y[i - 1] * dt
            m[i] = m[i - 1]
            v[i] = np.sqrt(v_y[i]**2 + v_x[i]**2)
            drag[i], rho[i] = [0,0]
            theta_gravity[i] = np.arctan2(d_x[i],R_e + d_y[i])
            a_x[i] = (-m[i] * g * np.sin(theta_gravity[i])) / m[i]
            a_y[i] = (-m[i] * g * np.cos(theta_gravity[i])) / m[i]
            SOS = SOS_calc(d_y[i])
            Mach[i] = v[i]/SOS
              
    return m, d_x, d_y, v_x, v_y, a_x, a_y, drag, rho, Mach, t, delta_vee_g_on_t, theta



if __name__ == "__main__":  # for testing

    ## Constants
    g = 9.81
    t_steps = 100000
    G = 6.673e-11
    M_e = 5.98e24
    R_e = 6.38e6
    coast_to_burn = 10

    
    ## Inputs
    Cd = 0.2
    m_dry = 300
    rocket_diameter = 0.5
    design_thrust = 50000
    design_Isp = 300
    target_altitude = 300e3
    starting_alt = 0
    turn_point = 5000
    stages = 2
    stage_h = [0,100000]
    stage_m_dry = np.array([m_dry,m_dry*0.2])
    stage_m_dot = np.array([1,0.1])*design_thrust/(design_Isp*g)
    
    
    ## Initial Estimates
    orbital_vel = np.sqrt(G*M_e/(R_e+target_altitude))
    delta_vee_drag = 200
    delta_vee_gravity = 2500
    
    delta_vee_req = orbital_vel + delta_vee_gravity + delta_vee_drag
    delta_vee_stage = np.array([delta_vee_req*1/3,delta_vee_req*2/3])
    mass_ratio = np.exp(delta_vee_stage/(design_Isp*g))
    
    ## Flow rates and thrust profiles (stages can be taken into account here? need mass decrease too though)
    thrust_profile = np.linspace(design_thrust,design_thrust,t_steps)/design_thrust
    
    ## Change to a smart loop or something
    stage_m_prop_2 = (mass_ratio[1]-1)*(stage_m_dry[1])
    stage_m_init_2 = stage_m_prop_2 + stage_m_dry[1]
    stage_m_prop_1 = (mass_ratio[0]-1)*(stage_m_init_2 + (stage_m_dry[0]-stage_m_dry[1]))
    stage_m_init_1 = stage_m_prop_1 + stage_m_dry[0] - stage_m_dry[1] + stage_m_init_2
    
    stage_m_init = np.array([stage_m_init_1,stage_m_init_2])
    stage_m_prop = np.array([stage_m_prop_1,stage_m_prop_2])
    
    t = np.linspace(0,1000,t_steps)

    m, d_x, d_y, v_x, v_y, a_x, a_y, drag, rho = initial_values(stage_m_dot[0], design_Isp, g, stage_m_init[0], starting_alt, t_steps, Cd, rocket_diameter,t)
    delta_vee_g_on_t = np.zeros(np.shape(v_x))
        
    for stage in range(1,stages+1):
        m, d_x, d_y, v_x, v_y, a_x, a_y, drag, rho, Mach, t, delta_vee_g_on_t, theta = launch_traj(t, m, v_x, v_y, d_x, d_y, drag, rho, a_x, a_y, g, Cd, rocket_diameter, turn_point, R_e, coast_to_burn, stage, stage_m_dry, stage_m_init, stage_m_prop, stage_m_dot, stage_h, design_Isp, thrust_profile, delta_vee_g_on_t)
        II = np.where(m == np.min(m[m!= 0]))[0][0]+1
        print(II)
        print(d_y[II-1])
        
    t_burn_end_ind = np.where(m == np.min(m[m!= 0]))[0][0]+1
    
    delta_vee_drag = np.trapz(drag[:t_burn_end_ind]/m[:t_burn_end_ind],x = t[:t_burn_end_ind])
    delta_vee_gravity = np.sum(delta_vee_g_on_t*t[1])

    plt.figure()
    plt.plot(t[1:],np.arctan2(v_y,v_x)[1:], t[1:], theta[1:])
    
    plt.figure()
    plt.plot(t,d_x,t,d_y,t,np.sqrt(d_x**2 + d_y**2))
    
    plt.figure()
    plt.plot(t,v_x,t,v_y,t,np.sqrt(v_x**2 + v_y**2))

    plt.figure()
    plt.plot(t,drag)
    
    plt.figure()
    plt.plot(t,a_x/g,t,a_y/g)
    
    ## At end of burn
    print('Burn Time = {}'.format(t[t_burn_end_ind]))
    print('Velocity (x,y) = ({},{})'.format(v_x[t_burn_end_ind], v_y[t_burn_end_ind]))
    print('Position (x,y) = ({},{})'.format(d_x[t_burn_end_ind], d_y[t_burn_end_ind]))
    print('Angle = {}'.format(np.arctan2(v_y[t_burn_end_ind], v_x[t_burn_end_ind])))
    
    
    
