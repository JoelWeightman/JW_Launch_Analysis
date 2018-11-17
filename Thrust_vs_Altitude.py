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
    
    for i,alt in enumerate(altitudes):
        
        ind = np.where(abs(alt-h) == np.min(abs(alt-h)))[0][0]
        if alt < h[ind]:
            ind -= 1

        if L[ind] == 0:
            P_var[i] = P[ind]*np.exp(-g*(alt-h[ind])/(R*T[ind]))
        else:
            P_var[i] = P[ind]*(T[ind]/(T[ind]+L[ind]*(alt-h[ind])))**(g/(R*L[ind]))
    
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

if __name__ == "__main__":
    
    ## Constants
    R_star = 8.314459848
    Molar_mass = 28.9645e-3
    R = R_star/Molar_mass
    gamma = 1.14
    P_SL = 101325
    T_SL = 288
    g_0 = 9.80665
    
    alt_steps = 10000
    
    m_dot_design = 1.19
    T_combustion = 4000
    P_comb = 500000
    
    altitudes = np.linspace(0,150e3,alt_steps)
    P_variation, altitudes = pressure_variation(altitudes,g_0,R)

    ## Bell
    P_design_bell = 0.02*P_comb
    design_altitude_bell_ind = np.where(abs(P_variation-P_design_bell) == np.min(abs(P_variation-P_design_bell)))[0][0]
    design_alt_bell = altitudes[design_altitude_bell_ind]
    
    A_ratio, Mj, Tj = area_ratio_calc(P_comb,P_design_bell,gamma,T_combustion)
    v_e = exhaust_velocity(gamma,R,Tj,Mj)
    bell_thrust_design = m_dot_design*v_e
    
    T_choked,rho_choked,c_choked,d_throat = choked_conditions(P_comb,T_combustion,R,gamma,m_dot_design)
    
    A_ratio, Mj, Tj = area_ratio_calc(P_comb,P_design_bell,gamma,T_combustion)
    A_exit = exit_area(A_ratio,d_throat)
    
    bell_thrust = thrust_output(P_variation,P_comb,gamma,P_design_bell,bell_thrust_design,A_exit)
    plt.plot(altitudes,bell_thrust)
    
    ## Aerospike
    P_design_aero = 0.02*P_comb
    design_altitude_aero_ind = np.where(abs(P_variation-P_design_aero) == np.min(abs(P_variation-P_design_aero)))[0][0]
    design_alt_aero = altitudes[design_altitude_aero_ind]
    
    A_ratio, Mj, Tj = area_ratio_calc(P_comb,P_design_aero,gamma,T_combustion)
    v_e = exhaust_velocity(gamma,R,Tj,Mj)
    thrust_design = m_dot_design*v_e
    T_choked,rho_choked,c_choked,d_throat = choked_conditions(P_comb,T_combustion,R,gamma,m_dot_design)
    
    A_ratio, Mj, Tj = area_ratio_calc(P_comb,P_variation,gamma,T_combustion)
    v_e = exhaust_velocity(gamma,R,Tj,Mj)
    aerospike_thrust = m_dot_design*v_e
    Isp = aerospike_thrust/(m_dot_design*g_0)

    plt.plot(altitudes,aerospike_thrust)
    plt.scatter(np.array([design_alt_aero,design_alt_bell]),np.array([thrust_design,thrust_design]),10,'k')
    
    
    
    