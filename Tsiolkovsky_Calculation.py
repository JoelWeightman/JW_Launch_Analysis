# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 13:44:27 2018

@author: jlwei
"""

import numpy as np
import matplotlib.pyplot as plt


def set_constants():
    
    g_0 = 9.81
    G_c = 6.673e-11
    M_e = 5.98e24
    R_e = 6.38e6
    
    R_star = 8.314459848
    Molar_mass = 28.9645e-3
    R_air = R_star/Molar_mass
    R_prop = 279.737
    
    t_steps = 10000
    
    return g_0, G_c, M_e, R_e, R_air, R_prop, t_steps
    
def set_rocket_props(accel_init, g_0, G_c, M_e, R_e, R_air, R_prop, engine):
    
    gamma = 1.138
    T_comb = 3885
    P_comb = 6e6
    
    target_altitude = 300e3
    
    orbital_vel = np.sqrt(G_c*M_e/(R_e+target_altitude))
    delta_vee_drag = 200
    delta_vee_gravity = 1300
    
    delta_vee_req = orbital_vel + delta_vee_gravity + delta_vee_drag
    Isp_design = 350
    m_payload_estimate_bell = 10
    
    m_init = m_payload_estimate_bell*np.exp(delta_vee_req/(Isp_design*g_0))
        
    altitude_design = 16.2e3
    thrust_design = 1#thrust_current(altitude_design, m_dot, P_comb, P_design, T_comb, gamma, R_air, R_prop, g_0, A_exit, 'bell')
    m_dot, A_exit, P_design = m_dot_design(thrust_design, altitude_design, P_comb, T_comb, gamma, R_air, R_prop, g_0)
    thrust_ratio = thrust_current(0, m_dot, P_comb, P_design, T_comb, gamma, R_air, R_prop, g_0, A_exit, 'bell')
        
    thrust_design = (1+accel_init)*m_init*g_0/thrust_ratio
    m_dot, A_exit, P_design = m_dot_design(thrust_design, altitude_design, P_comb, T_comb, gamma, R_air, R_prop, g_0)
    thrust_SL = thrust_current(0, m_dot, P_comb, P_design, T_comb, gamma, R_air, R_prop, g_0, A_exit, 'bell')
    
    rocket_diam = 0.1
    Cd = 0.2
      
    return delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, P_design, A_exit, orbital_vel
    
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

def current_pressure(alt,g_0,R):
    
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
            P_var = P[ind]*np.exp(-g_0*(alt-h[ind])/(R*T[ind]))
            T_var = T[ind]
        else:
            P_var = P[ind]*(T[ind]/(T[ind]+L[ind]*(alt-h[ind])))**(g_0/(R*L[ind]))
            T_var = T[ind]+L[ind]*(alt-h[ind])
            
        rho_var = P_var/(R*T_var)
        return P_var, T_var, rho_var

def thrust_current(alt,m_dot,P_chamber,P_design,T_chamber,gamma,R_air,R_prop,g_0,A_exit,engine):
    
    P_alt,T_alt,rho_alt = current_pressure(alt,g_0,R_air)
     
    
    if engine == 'bell':
        
        AR, M, T = area_ratio_calc(P_chamber,P_design,gamma,T_chamber)
    
        v_e = exhaust_velocity(gamma,R_prop,T,M)
      
        thrust = m_dot*v_e + A_exit*(P_design-P_alt)
        
    elif engine == 'spike':
        
        AR, M, T = area_ratio_calc(P_chamber,P_alt,gamma,T_chamber)
    
        v_e = exhaust_velocity(gamma,R_prop,T,M)
               
        thrust = m_dot*v_e
    
    return thrust

def m_dot_design(thrust_design,alt,P_chamber,T_chamber,gamma,R_air,R_prop,g_0):

    P_alt,T_alt,rho_alt = current_pressure(alt,g_0,R_air)
    
    AR, M, T = area_ratio_calc(P_chamber,P_alt,gamma,T_chamber)
    
    v_e = exhaust_velocity(gamma,R_prop,T,M)
    
    m_dot = thrust_design/v_e
    
    rho = P_alt/(T*R_prop)
    
    A_exit = m_dot/(rho*v_e)
   
    return m_dot, A_exit, P_alt

def generate_path(target_altitude):
    
    h_pre_turn = 2e3
    alt_steps = 1000

    y_end = target_altitude
    x_end = target_altitude/3
    
    y = np.concatenate((np.array([0]),np.linspace(h_pre_turn,y_end,alt_steps)))
    x_circ = -(1-((y[1:]-h_pre_turn)/(y_end-h_pre_turn))**2)**(1/2)*x_end+x_end
    x = np.concatenate((np.array([0]),x_circ))
    
    distance = np.sqrt((x[1:]-x[:-1])**2+(y[1:]-y[:-1])**2)
    
    path_distance = np.concatenate((np.array([0]),np.cumsum(distance)))
    path_angle = np.arctan2((y[1:]-y[:-1]),(x[1:]-x[:-1]))
    path_angle = np.concatenate((path_angle,np.array([0])))
    
    path = np.zeros((alt_steps+1,3))
    
    path[:,0] = y
    path[:,1] = path_distance
    path[:,2] = path_angle
    
    plt.figure()
    plt.plot(x,y)
    plt.axis('equal')
    
    
    return path

def get_drag_angle(distance,velocity,path,Cd,rocket_diam):
    
    altitude_current = np.interp(distance,path[:,1],path[:,0])
    phi = np.interp(distance,path[:,1],path[:,2])
    
    drag = drag_calc(velocity,altitude_current,Cd,rocket_diam)
     
    return drag, phi, altitude_current
    
def drag_calc(v,alt,Cd,rocket_diam):

    density = density_calc(alt)
    
    Area = rocket_diam**2*np.pi

    drag = 0.5 * density * v * v * Cd * Area

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
    
    

def Tsiolkovsky(g_0, delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, t_steps, P_design, A_exit, engine, path, orbital_vel):
    
    eps = m_init*1e-6
    
    t_burn = (m_init-eps)/m_dot
    
    m = np.linspace(m_init,eps,t_steps)
    t = np.linspace(0,t_burn,t_steps)
    dt = t[1]-t[0]
    
    m = np.zeros(np.shape(t)[0]*2)
    m[0] = m_init
    
    delta_vee_seg = np.zeros(np.shape(t)[0]*2)
    v_e = np.zeros(np.shape(t)[0]*2)
    m_dot_spike = np.zeros(np.shape(t)[0]*2)
    thrust_bell = np.zeros(np.shape(t)[0]*2)
    thrust_spike = np.zeros(np.shape(t)[0]*2)
    drag_bell = np.zeros(np.shape(t)[0]*2)
    velocity_bell = np.zeros(np.shape(t)[0]*2)
    distance_bell = np.zeros(np.shape(t)[0]*2)
    altitude_bell = np.zeros(np.shape(t)[0]*2)
    drag_spike = np.zeros(np.shape(t)[0]*2)
    velocity_spike = np.zeros(np.shape(t)[0]*2)
    distance_spike = np.zeros(np.shape(t)[0]*2)
    altitude_spike = np.zeros(np.shape(t)[0]*2)
    
    i = 0
    
    if engine == 'bell':
        while m[i] >= 0:
            i += 1
            
            drag_bell[i],phi,altitude_bell[i] = get_drag_angle(distance_bell[i-1],velocity_bell[i-1],path,Cd,rocket_diam)
            thrust_bell[i] = thrust_current(altitude_bell[i],m_dot,P_comb, P_design, T_comb, gamma, R_air, R_prop, g_0, A_exit, 'bell')
            
            m[i] = m[i-1] - m_dot*dt
            if m[i] < 0:
                break
            
            v_prime = (thrust_bell[i] - drag_bell[i])/m[i] - g_0*np.sin(phi)
            velocity_bell[i] = velocity_bell[i-1] + v_prime*dt
            distance_bell[i] = distance_bell[i-1] + velocity_bell[i]*dt
            
            v_e[i] = thrust_bell[i]/m_dot
            
            delta_vee_seg[i] = v_e[i]*np.log(m[i-1]/m[i])
            
            if altitude_bell[i] >= target_altitude:
                break
            
            
        delta_vee = np.cumsum(delta_vee_seg)
        ind = np.where(abs(velocity_bell-orbital_vel) == np.min(abs(velocity_bell-orbital_vel)))[0][0]
#        ind = np.where(abs(delta_vee-delta_vee_req) == np.min(abs(delta_vee-delta_vee_req)))[0][0]
#        ind = np.where(abs(altitude_bell-target_altitude) == np.min(abs(altitude_bell-target_altitude)))[0][0]
        m_payload = m[ind]
                
    elif engine == 'spike':
        while m[i] >= 0:
            i += 1
            
            drag_spike[i],phi,altitude_spike[i] = get_drag_angle(distance_spike[i-1],velocity_spike[i-1],path,Cd,rocket_diam)
            thrust_bell[i] = thrust_current(altitude_spike[i],m_dot,P_comb, P_design, T_comb, gamma, R_air, R_prop, g_0, A_exit, 'bell')
            thrust_spike[i] = thrust_current(altitude_spike[i],m_dot,P_comb, P_design, T_comb, gamma, R_air, R_prop, g_0, A_exit, 'spike')
            
            m_dot_spike[i] = m_dot#*thrust_bell[i]/thrust_spike[i]
            m[i] = m[i-1] - m_dot_spike[i]*dt
            
            if m[i] < 0:
                break
            
            v_prime = (thrust_spike[i] - drag_spike[i])/m[i] - g_0*np.sin(phi)
            velocity_spike[i] = velocity_spike[i-1] + v_prime*dt
            distance_spike[i] = distance_spike[i-1] + velocity_spike[i]*dt
            
            v_e[i] = thrust_spike[i]/m_dot_spike[i]
            
            if altitude_spike[i] >= target_altitude:
                break
            
            delta_vee_seg[i] = v_e[i]*np.log(m[i-1]/(m[i]))
            
        delta_vee = np.cumsum(delta_vee_seg)
        ind = np.where(abs(velocity_spike-orbital_vel) == np.min(abs(velocity_spike-orbital_vel)))[0][0]
#        ind = np.where(abs(delta_vee-delta_vee_req) == np.min(abs(delta_vee-delta_vee_req)))[0][0]
#        ind = np.where(abs(altitude_spike-target_altitude) == np.min(abs(altitude_spike-target_altitude)))[0][0]
        m_payload = m[ind]
        
    return m_payload, v_e, delta_vee, thrust_bell, thrust_spike, m, drag_bell, drag_spike, velocity_bell, velocity_spike, altitude_bell, altitude_spike, m_dot_spike
    
    
if __name__ == "__main__":
    
    plt.close('all')
    accel_init = 1.00 ## in gees
    engine = 'bell'
    
    g_0, G_c, M_e, R_e, R_air, R_prop, t_steps = set_constants()
    delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, P_design, A_exit, orbital_vel = set_rocket_props(accel_init, g_0, G_c, M_e, R_e, R_air, R_prop, engine)
    
    path = generate_path(target_altitude)
    
    m_payload_bell, v_e, delta_vee_bell, thrust_bell, temp, m_bell, drag_bell, temp, velocity_bell, temp, altitude_bell, temp, temp = Tsiolkovsky(g_0,delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, t_steps, P_design, A_exit, engine, path, orbital_vel)

    
    engine = 'spike'
    
    delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, P_design, A_exit, orbital_vel = set_rocket_props(accel_init, g_0, G_c, M_e, R_e, R_air, R_prop, engine)
    
    m_payload_spike, v_e, delta_vee_spike, temp, thrust_spike, m_spike, temp, drag_spike, temp, velocity_spike, temp, altitude_spike, m_dot_spike = Tsiolkovsky(g_0,delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, t_steps, P_design, A_exit, engine, path, orbital_vel)
    print('Bell Max Payload Percentage = %3.3f' % (m_payload_bell/m_init*100))
    print('Spike Max Payload Percentage = %3.3f' % (m_payload_spike/m_init*100))
    
    ind_bell = np.where(abs(delta_vee_bell-delta_vee_req) == np.min(abs(delta_vee_bell-delta_vee_req)))[0][0]
    ind_spike = np.where(abs(delta_vee_spike-delta_vee_req) == np.min(abs(delta_vee_spike-delta_vee_req)))[0][0]
    
    
    ind_bell = np.where(abs(velocity_bell-orbital_vel) == np.min(abs(velocity_bell-orbital_vel)))[0][0]
    ind_spike = np.where(abs(velocity_spike-orbital_vel) == np.min(abs(velocity_spike-orbital_vel)))[0][0]

    
    
    print('Bell delta vee = %3.3f' % delta_vee_bell[ind_bell])
    print('Spike delta vee = %3.3f' % delta_vee_spike[ind_spike])
    