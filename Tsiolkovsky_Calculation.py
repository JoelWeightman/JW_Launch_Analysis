# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 13:44:27 2018

@author: jlwei
"""

import numpy as np


def set_constants():
    
    g_0 = 9.81
    G_c = 6.673e-11
    M_e = 5.98e24
    R_e = 6.38e6
    
    R_star = 8.314459848
    Molar_mass = 28.9645e-3
    R_air = R_star/Molar_mass
    R_prop = 279.737
    
    t_steps = 1000
    
    return g_0, G_c, M_e, R_e, R_air, R_prop, t_steps
    
def set_rocket_props(accel_init, g_0, G_c, M_e, R_e, R_air, R_prop, engine):
    
    gamma = 1.138
    T_comb = 3885
    P_comb = 6e6
        
    altitude_design = 16.2e3
    Isp_design = 300
    thrust_design = 30000
    m_dot, A_exit, P_design = m_dot_design(thrust_design, altitude_design, P_comb, T_comb, gamma, R_air, R_prop, g_0)
    
    thrust_SL = thrust_current(0, m_dot, P_comb, P_design, T_comb, gamma, R_air, R_prop, g_0, A_exit, 'bell')
    rocket_diam = 0.5
    Cd = 0.2
    target_altitude = 300e3
    
    orbital_vel = np.sqrt(G_c*M_e/(R_e+target_altitude))
    delta_vee_drag = 150
    delta_vee_gravity = 1500
    
    delta_vee_req = orbital_vel + delta_vee_gravity + delta_vee_drag
    
    m_init = thrust_SL/((1+accel_init)*g_0)
      
    return delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, P_design, A_exit
    
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

def thrust_current(alt,m_dot,P_chamber,P_design,T_chamber,gamma,R_air,R_prop,g,A_exit,engine):
    
    P_alt,T_alt,rho_alt = current_pressure(alt,g,R_air)
    
    if engine == 'bell':
        
        AR, M, T = area_ratio_calc(P_chamber,P_design,gamma,T_chamber)
    
        v_e = exhaust_velocity(gamma,R_prop,T,M)
      
        thrust = m_dot*v_e + A_exit*(P_design-P_alt)
        
    elif engine == 'spike':
        
        AR, M, T = area_ratio_calc(P_chamber,P_alt,gamma,T_chamber)
    
        v_e = exhaust_velocity(gamma,R_prop,T,M)
               
        thrust = m_dot*v_e
    
    return thrust

def m_dot_design(thrust_design,alt,P_chamber,T_chamber,gamma,R_air,R_prop,g):

    P_alt,T_alt,rho_alt = current_pressure(alt,g,R_air)
    
    AR, M, T = area_ratio_calc(P_chamber,P_alt,gamma,T_chamber)
    
    v_e = exhaust_velocity(gamma,R_prop,T,M)
        
    m_dot = thrust_design/v_e
    
    rho = P_alt/(T*R_prop)
    
    A_exit = m_dot/(rho*v_e)
    
    return m_dot, A_exit, P_alt

def altitude_current(time, target_altitude, v):
    
    h_pre_turn = 2e3
    phi_init = 89
    phi_init *= np.pi/180
    
    
    return h_pre_turn
    
def Tsiolkovsky(g,delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, t_steps, P_design, A_exit, engine):
    
    eps = m_init*1e-6
    
    t_burn = (m_init-eps)/m_dot
    
    m = np.linspace(m_init,eps,t_steps)
    t = np.linspace(0,t_burn,t_steps)
    alt = np.linspace(0,target_altitude,t_steps)
    dt = t[1]-t[0]
    
    delta_vee_seg = np.zeros(np.shape(m)[0])
    v_e = np.zeros(np.shape(m)[0])
    m_dot_spike = np.zeros(np.shape(m)[0])
    thrust_bell = np.zeros(np.shape(m)[0])
    thrust_spike = np.zeros(np.shape(m)[0])
    
    
    if engine == 'bell':
        for i in range(1,np.size(m)):
            
            thrust_bell[i] = thrust_current(alt[i],m_dot,P_comb, P_design, T_comb, gamma, R_air, R_prop, g, A_exit, 'bell')
            v_e[i] = thrust_bell[i]/m_dot
            
            delta_vee_seg[i] = v_e[i]*np.log(m[i-1]/m[i])
            
        delta_vee = np.cumsum(delta_vee_seg)
        ind = np.where(abs(delta_vee-delta_vee_req) == np.min(abs(delta_vee-delta_vee_req)))[0][0]
        m_payload = m[ind]
                
    elif engine == 'spike':
        for i in range(1,np.size(m)):
            
            thrust_bell[i] = thrust_current(alt[i],m_dot,P_comb, P_design, T_comb, gamma, R_air, R_prop, g, A_exit, 'bell')
            thrust_spike[i] = thrust_current(alt[i],m_dot,P_comb, P_design, T_comb, gamma, R_air, R_prop, g, A_exit, 'spike')
            m_dot_spike[i] = m_dot*thrust_bell[i]/thrust_spike[i]
            v_e[i] = thrust_spike[i]/m_dot_spike[i]
            m[i] = m[i-1] - m_dot_spike[i]*dt
            if m[i] < 0:
                break
            
            delta_vee_seg[i] = v_e[i]*np.log(m[i-1]/(m[i]))
            
        delta_vee = np.cumsum(delta_vee_seg)
        ind = np.where(abs(delta_vee-delta_vee_req) == np.min(abs(delta_vee-delta_vee_req)))[0][0]
        m_payload = m[ind]
        
    return m_payload, thrust, v_e, delta_vee, thrust_bell, thrust_spike, m
    
    
if __name__ == "__main__":
    
    accel_init = 0.5 ## in gees
    engine = 'bell'
    
    g_0, G_c, M_e, R_e, R_air, R_prop, t_steps = set_constants()
    delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, P_design, A_exit = set_rocket_props(accel_init, g_0, G_c, M_e, R_e, R_air, R_prop, engine)
    
    m_payload_bell, thrust, v_e, delta_vee, thrust_bell, thrust_spike, m = Tsiolkovsky(g,delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, t_steps, P_design, A_exit, engine)

    
    engine = 'spike'
    
    g_0, G_c, M_e, R_e, R_air, R_prop, t_steps = set_constants()
    delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, P_design, A_exit = set_rocket_props(accel_init, g_0, G_c, M_e, R_e, R_air, R_prop, engine)
    
    m_payload_spike, thrust, v_e, delta_vee, thrust_bell, thrust_spike, m = Tsiolkovsky(g,delta_vee_req, m_init, gamma, T_comb, P_comb, Isp_design, thrust_design, thrust_SL, rocket_diam, Cd, target_altitude, altitude_design, m_dot, t_steps, P_design, A_exit, engine)
    print('Bell Max Payload = %3.3fkg' % m_payload_bell)
    print('Spike Max Payload = %3.3fkg' % m_payload_spike)