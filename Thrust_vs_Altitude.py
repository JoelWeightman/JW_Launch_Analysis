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
    
#def pressure_variation(alt_steps,altitudes,P_SL,g):
#    
#    densities = altitude_variation_rho(altitudes)
#    P = np.zeros(np.shape(altitudes))
#    d_alt = altitudes[1]-altitudes[0]
#    
#    P[0] = P_SL
#    
#    for i in range(1,alt_steps):
#        P[i] = P[i-1] - densities[i-1]*g*d_alt
#
##    P[P<1] = 1
#    
#    alt_steps_needed = alt_steps/1000
#    
#    return P[::alt_steps_needed], altitudes[::alt_steps_needed]

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

def altitude_variation_P(altitudes):
    
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
       4.50000000e+01, 1.50000000e+02, 5.00000000e+01, 8.00000000e+01,
       5.50000000e+01, 6.00000000e+01, 6.00000000e+01, 2.00000000e+01,
       6.50000000e+01, 1.50000000e+01, 7.00000000e+01, 5.00000000e+00,
       8.00000000e+01, 1.00000000e+00, 9.00000000e+01, 1.00000000e+00,
       1.00000000e+02, 1.00000000e+00, 1.10000000e+02, 1.00000000e+00])
    
    alt = Temporary_pressure[0::2] * 10**3
    pressure = Temporary_pressure[1::2]

    pressure_var = np.interp(altitudes, alt, pressure)
    
    return pressure_var

def altitude_variation_rho(altitudes):
       
    Temporary_density = np.array([
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

    alt = Temporary_density[0::2] * 10**3
    density = Temporary_density[1::2]

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
    R_star = 8.314459848
    Molar_mass = 28.9645e-3
    R = R_star/Molar_mass
    gamma = 1.15
    P_SL = 101325
    T_SL = 288
    g_0 = 9.80665
    
    alt_steps = 10000
    
    m_dot_design = 10
    T_combustion = 3500
    P_comb = 70*P_SL
    
    altitudes = np.linspace(0,150e3,alt_steps)
    P_variation, altitudes = pressure_variation(altitudes,g_0,R)

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


#    thrust = thrust_output(P_variation,P_comb,gamma,P_design,aerospike_thrust,A_exit,True)
    plt.plot(altitudes,aerospike_thrust)
    plt.scatter(np.array([design_alt_aero,design_alt_bell]),np.array([thrust_design,thrust_design]),10,'k')
    
    
    
    