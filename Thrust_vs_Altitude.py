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
        
        if P_variation[0] > P_design*2*3:
            ind = np.where(P_variation[P_variation > P_design*2*3])[0][-1]
        
            thrust[:ind] = thrust[ind+1]
     
    return thrust
    
def pressure_variation(alt_steps,altitudes,P_SL,g):
    
    densities = altitude_variation(altitudes)
    P = np.zeros(np.shape(altitudes))
    d_alt = altitudes[1]-altitudes[0]
    
    P[0] = P_SL
    
    for i in range(1,alt_steps):
        P[i] = P[i-1] - densities[i-1]*g*d_alt

    return P[::100], altitudes[::100]

def altitude_variation(altitudes):
       
    
    Temporary_pressure = np.array([
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

    alt = Temporary_pressure[0::2] * 10**3
    density = Temporary_pressure[1::2]

    rho_variation = np.interp(altitudes, alt, density)
    
    return rho_variation

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
    R = 287
    gamma = 1.15
    P_SL = 101325
    T_SL = 288
    g_0 = 9.80665
    
    alt_steps = 1000000
    
    m_dot_design = 10
    T_combustion = 3500
    P_comb = 70*P_SL
    
    altitudes = np.linspace(0,100e3,alt_steps)
    P_variation, altitudes = pressure_variation(alt_steps,altitudes,P_SL,g_0)

    ## Bell
    P_design_bell = 0.01*P_SL
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
    P_design_aero = 0.01*P_SL
    design_altitude_aero_ind = np.where(abs(P_variation-P_design_aero) == np.min(abs(P_variation-P_design_aero)))[0][0]
    design_alt_aero = altitudes[design_altitude_aero_ind]
    
    A_ratio, Mj, Tj = area_ratio_calc(P_comb,P_design_aero,gamma,T_combustion)
    v_e = exhaust_velocity(gamma,R,Tj,Mj)
    thrust_design = m_dot_design*v_e
    T_choked,rho_choked,c_choked,d_throat = choked_conditions(P_comb,T_combustion,R,gamma,m_dot_design)
    
    A_ratio, Mj, Tj = area_ratio_calc(P_comb,P_variation,gamma,T_combustion)
    v_e = exhaust_velocity(gamma,R,Tj,Mj)
    aerospike_thrust = m_dot_design*v_e


##    thrust = thrust_output(P_variation,P_comb,gamma,P_design,aerospike_thrust,A_exit,True)
#    plt.plot(altitudes,aerospike_thrust)
#    plt.scatter(np.array([design_alt_aero,design_alt_bell]),np.array([thrust_design,thrust_design]),10,'k')
    
    
    
    