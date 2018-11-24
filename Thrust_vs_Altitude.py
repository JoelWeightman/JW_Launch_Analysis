# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:56:05 2018

@author: JL
"""

import numpy as np
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

def thrust_output(P_variation,P_comb,gamma,P_design,thrust_design,A_exit,aerospike = False):
    
    if aerospike == True:
        thrust = thrust_design
    else:
        P_difference = P_design - P_variation  
        thrust = thrust_design + P_difference*A_exit
        
#        if P_variation[0] > P_design*2*3:
#            ind = np.where(P_variation[P_variation > P_design*2*3])[0][-1]
#        
#            thrust[:ind] = thrust[ind+1]
     
    return thrust

def pressure_variation(altitudes,g,R):
    
    h = np.array([0,11e3,20e3,32e3,47e3,51e3,71e3])
    P = np.array([101325.0,22632.1,5474.89,868.02,110.91,66.94,3.96])
    T = np.array([288.15,216.65,216.65,228.65,270.65,270.65,214.65])
    L = np.array([-0.0065,0,0.001,0.0028,0,-0.0028,-0.002])
    
    P_var = np.zeros(np.shape(altitudes))
    
    if np.size(altitudes) > 1:
        for i,alt in enumerate(altitudes):
            
            ind = np.where(abs(alt-h) == np.min(abs(alt-h)))[0][0]
            if alt < h[ind]:
                ind -= 1
    
            if L[ind] == 0:
                P_var[i] = P[ind]*np.exp(-g*(alt-h[ind])/(R*T[ind]))
            else:
                P_var[i] = P[ind]*(T[ind]/(T[ind]+L[ind]*(alt-h[ind])))**(g/(R*L[ind]))
    elif np.size(altitudes) == 1:
               
        ind = np.where(abs(altitudes-h) == np.min(abs(altitudes-h)))[0][0]
        if altitudes < h[ind]:
            ind -= 1

        if L[ind] == 0:
            P_var = P[ind]*np.exp(-g*(altitudes-h[ind])/(R*T[ind]))
        else:
            P_var = P[ind]*(T[ind]/(T[ind]+L[ind]*(altitudes-h[ind])))**(g/(R*L[ind]))
            
    return P_var, altitudes

def choked_conditions(P1,T1,R,gam,m_dot):
    
    npr_crit = 1/(2/(gam+1))**(gam/(gam-1))
    
    rho_up = P1/(R*T1)
    
    T_choked = T1*(1/npr_crit)**((gam-1)/gam)
    c_choked = np.sqrt(R*gam*T_choked)
    rho_choked = rho_up*(1/npr_crit)**(1/gam)
    
    A = m_dot/(rho_choked*c_choked)
    
    d_throat = np.sqrt(A/np.pi)*2
    
    return T_choked,rho_choked,c_choked,d_throat

def set_constants(condition):
    
    R_star = 8.314459848  
    P_SL = 101325
    T_SL = 288
    g_0 = 9.80665
    m_dot_design = 1.00
    
    alt_steps = 10000
    max_alt = 100e3
    altitudes = np.linspace(0,max_alt,alt_steps)
    
    if condition == 'cold':
        Molar_mass = 28.9645e-3
        R = R_star/Molar_mass
        T_comb = 300
        P_comb = 500000
        gamma = 1.4
    elif condition == 'hot':
        Molar_mass = 29.7226e-3
        R = R_star/Molar_mass
        T_comb = 3885
        P_comb = 6000000
        gamma = 1.138
    
    return R, T_comb, P_comb, T_SL, P_SL, g_0, m_dot_design, altitudes, gamma
    
def calculate_bell(P_design_bell, P_variation, P_comb, gamma, T_comb, R):
    
    ## Bell
    design_altitude_bell_ind = np.where(abs(P_variation-P_design_bell) == np.min(abs(P_variation-P_design_bell)))[0][0]
    design_alt_bell = altitudes[design_altitude_bell_ind]
    
    A_ratio, Mj, Tj = area_ratio_calc(P_comb,P_design_bell,gamma,T_comb)
    v_e = exhaust_velocity(gamma,R,Tj,Mj)
    thrust_design = m_dot_design*v_e
    
    T_choked,rho_choked,c_choked,d_throat = choked_conditions(P_comb,T_comb,R,gamma,m_dot_design)
    
    A_ratio, Mj, Tj = area_ratio_calc(P_comb,P_design_bell,gamma,T_comb)
    A_exit = exit_area(A_ratio,d_throat)
    
    bell_thrust = thrust_output(P_variation,P_comb,gamma,P_design_bell,thrust_design,A_exit)
    
    return bell_thrust, design_alt_bell, thrust_design

def calculate_spike(P_design_aero, P_variation, P_comb, gamma, T_comb, R):
    ## Aerospike
    
    design_altitude_aero_ind = np.where(abs(P_variation-P_design_aero) == np.min(abs(P_variation-P_design_aero)))[0][0]
    design_alt_aero = altitudes[design_altitude_aero_ind]
    
    A_ratio, Mj, Tj = area_ratio_calc(P_comb,P_design_aero,gamma,T_comb)
    v_e = exhaust_velocity(gamma,R,Tj,Mj)
    thrust_design = m_dot_design*v_e
    T_choked,rho_choked,c_choked,d_throat = choked_conditions(P_comb,T_comb,R,gamma,m_dot_design)
    
    A_ratio, Mj, Tj = area_ratio_calc(P_comb,P_variation,gamma,T_comb)
    v_e = exhaust_velocity(gamma,R,Tj,Mj)
    aero_thrust = m_dot_design*v_e
    
    return aero_thrust, design_alt_aero, thrust_design

if __name__ == "__main__":
  
    condition = 'hot'
    P_design_bell_alt = 16.2e3
    P_design_aero_alt = 20.2e3
    
    R, T_comb, P_comb, T_SL, P_SL, g_0, m_dot_design, altitudes, gamma = set_constants(condition)
    
    P_variation, altitudes = pressure_variation(altitudes,g_0,R)
    P_design_bell, alt_temp = pressure_variation(np.array(P_design_bell_alt),g_0,R)
    bell_thrust, design_alt_bell, thrust_design_bell = calculate_bell(P_design_bell, P_variation, P_comb, gamma, T_comb, R)
     
    P_design_aero, alt_temp = pressure_variation(np.array(P_design_aero_alt),g_0,R)
    aerospike_thrust, design_alt_aero, thrust_design_aero = calculate_spike(P_design_aero, P_variation, P_comb, gamma, T_comb, R)
       
    Isp_bell = bell_thrust/(m_dot_design*g_0)
    Isp_aero = aerospike_thrust/(m_dot_design*g_0)
    
    plt.plot(altitudes,bell_thrust)
    plt.plot(altitudes,aerospike_thrust)
    plt.scatter(np.array([design_alt_aero,design_alt_bell]),np.array([thrust_design_aero,thrust_design_bell]),10,'k')
    
    
    
    