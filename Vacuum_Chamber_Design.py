# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 17:14:58 2018

@author: JL
"""
""" Vacuum Chamber Design """

import numpy as np
import matplotlib.pyplot as plt

def chamber_volume(D,L):
    
    V = np.pi*(D/2)**2*L
    
    return V

def chamber_mass(V,P,T):
    
    R = 287
    rho = P/(R*T)
    
    m = V*rho
    
    return m

def nozzle_m_dot(d,c,rho):
    
    A = np.pi*(d/2)**2
    m_dot = A*rho*c
     
    return m_dot

def aerospike_sizing(d,h):
    
    A = np.pi*(d/2)**2
    
    r_spike = (A-np.pi*h**2)/(2*np.pi*h)
    
    d_spike = r_spike*2
    
    return d_spike

def choked_conditions(P1,T1):
    
    R = 287
    gam = 1.4
    npr_crit = 1/(2/(gam+1))**(gam/(gam-1))
    
    rho_up = P1/(R*T1)
    
    T_choked = T1*(1/npr_crit)**((gam-1)/gam)
    c_choked = np.sqrt(R*gam*T_choked)
    rho_choked = rho_up*(1/npr_crit)**(1/gam)
    
    return T_choked,rho_choked,c_choked

def pressure_v_time(m_dot,run_time,m_init,V,T,t_steps):

    R = 287    
    
    m = np.linspace(m_init,m_dot*run_time,t_steps)
    
    rho = m/V
    
    P = rho*R*T
    
    return P

def altitude_variation(P_variation,T):
       
    R = 287    
    
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

    alt = np.flip(Temporary_pressure[0::2] * 10**3,axis = 0)
    pressure = np.flip(Temporary_pressure[1::2]*R*T,axis = 0)

    altitude = np.interp(P_variation, pressure, alt)
    
    return altitude

if __name__ == "__main__":
    
    
    ## Chamber Sizing (m)
    d_vac = 0.5
    L = 6
    
    ## Nozzle Sizing (m)
    d_throat = 0.01 #(Effective d_throat for aerospike)
    h_annulus = 0.001
    
    ## Pressures (Pa)
    P_atm = 101325
    P_vac = 100
    P_final = 1*101325
    P_up = P_atm*5
    
    P_ratio_design = P_up/3262.1
    
    ## others
    T = 288
    t_steps = 1000
    rho_current = 0.000977525
    
    ## Vac Pump Specs
    m_dot_vac_out = 100 ## (m3/hr)
    m_dot_vac_out = m_dot_vac_out/(60*60)*rho_current ## (kg/s)
     
    d_spike = aerospike_sizing(d_throat,h_annulus)
    
    T_choked,rho_choked,c_choked = choked_conditions(P_up,T)

    V = chamber_volume(d_vac,L)

    m_vac = chamber_mass(V,P_vac,T)
    m_final = chamber_mass(V,P_final,T)
    
    m_dot = nozzle_m_dot(d_throat,c_choked,rho_choked)
    
    run_time = (m_final-m_vac)/(m_dot-m_dot_vac_out)
    
    t = np.linspace(0,run_time,t_steps)
    
    P_variation = pressure_v_time(m_dot,run_time,m_vac,V,T,t_steps)
    
    altitude = altitude_variation(P_variation,T)
    
    plt.figure()
    plt.plot(t,altitude)
    
    plt.figure()
    plt.plot(t,(P_up/P_variation)/P_ratio_design)

    