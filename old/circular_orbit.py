# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:33:52 2018

@author: JL
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:57:53 2018

@author: JL
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def thrust_current(alt,thrust_design):
    
    thrust = thrust_design
    
    return thrust

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

def model_burn(variables,t,G,M_e,R_e,m_dot,thrust_design,rocket_diam,Cd):
    
    v,phi,r,theta,m = variables
    
    R_current = r
    altitude = R_current - R_e
    T = thrust_current(altitude,thrust_design)
    D = drag_current(v,Cd,altitude,rocket_diam)
    
    v_prime = (T - D)/m - (G*M_e/R_current**2 - v**2/R_current)*np.sin(phi)
    phi_prime = -1/v*(G*M_e/R_current**2)*np.cos(phi) + v*np.cos(phi)/R_current
    r_prime = v*np.sin(phi)
    theta_prime = v/R_current*np.cos(phi)
    m_prime = -m_dot
    
    return v_prime, phi_prime, r_prime, theta_prime, m_prime
    
    
if __name__ == "__main__":
    
#    plt.close('all')
    
    g = 9.81
    G = 6.673e-11
    M_e = 5.98e24
    R_e = 6.38e6
    t_steps = 1000
    
    Isp_design = 300
    thrust_design = 50000
    rocket_diam = 0.5
    Cd = 0.2
    m_dry = 500
    stage_m_dry = np.array([m_dry,m_dry*0.1])
    stage_m_dot = np.array([1,0.1])*thrust_design/(Isp_design*g)
    
    event_alt = 1500
    GT_angle = 89*np.pi/180
    
    target_altitude = 300e3
    orbital_period = 2*np.pi*np.sqrt((R_e+target_altitude)**3/(G*M_e))
    
    orbital_vel = np.sqrt(G*M_e/(R_e+target_altitude))
   
    mass = 100
     
    t = np.linspace(0,orbital_period,t_steps)
    init_conds = [orbital_vel, 0, R_e+target_altitude, np.pi/2, mass]
    trajectory = odeint(model_burn, init_conds, t, args=(G,M_e,R_e,0,0,rocket_diam,Cd))
    
    [v,phi,r,theta,m] = np.transpose(trajectory)

    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    plt.figure()
    plt.plot(x/1000,(y-R_e)/1000)
    plt.axis('equal')
    plt.xlabel('Downrange (km)')
    plt.ylabel('Altitude (km)')
    
    altitudes = np.sqrt(x**2 + y**2) - R_e
        
    plt.figure()
    plt.plot(t,altitudes/1000)
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    
    plt.figure()
    plt.plot(t,phi*180/np.pi)
    plt.xlabel('Time (s)')
    plt.ylabel('Flight Angle (deg)')
    
    plt.figure()
    plt.plot(t,theta*180/np.pi)
    plt.xlabel('Time (s)')
    plt.ylabel('Central Angle (deg)')
    
    plt.figure()
    plt.plot(t,v/1000)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (km/s)')
    
#    print('Initial Acceleration = {}g,{}g'.format((stage_m_dot[0]*Isp_design*g/(g*stage_m_init[0])-1),(stage_m_dot[1]*Isp_design*g/(g*stage_m_init[1])-1)))
#    print('Final Angle = {}'.format(phi[p2*2]*180/np.pi))
#    print('Final Position Tangent = {}'.format(np.arctan2(y,x)[-1]-np.pi/2))
#    print('Final Altitude = {}'.format((np.sqrt(x**2 + y**2)-R_e)[-1]))
#    print('Final Velocity = {}'.format(v[-1]))
    

