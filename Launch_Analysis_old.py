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

def rocket_mass_calc(phi, g, m_dry, thrust, m_dot):

    m_init = thrust / (phi * g)
    m_prop = m_init - m_dry
    MR = m_init / m_dry
    t_burn = m_prop / m_dot

    return m_init, m_prop, MR, t_burn


def drag_current(V, Cd, d, rocket_diameter):

    Area = (rocket_diameter/2)**2*np.pi
    rho_curr = density_calc(d)
    drag = 0.5 * rho_curr * V * V * Cd * Area
#    print(rho_curr, V, Area)
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

def initial_values(phi, g, m_init, t_steps, Cd, rocket_diameter,t,d_init = 0):
    m = np.zeros(np.shape(t))
    d = np.zeros(np.shape(t))
    v = np.zeros(np.shape(t))
    a = np.zeros(np.shape(t))
    drag = np.zeros(np.shape(t))
    rho = np.zeros(np.shape(t))

    m[0] = m_init
    d[0] = d_init
    v[0] = 0
    a[0] = (phi - 1) * g
    drag[0], rho[0] = drag_current(v[0], Cd, d[0], rocket_diameter)

    return m, d, v, drag, a, rho


def ballistic_traj(t_1, t_2, v, d, m, drag, rho, a, thrust, g, m_dot, Cd, rocket_diameter):

    dt_1 = t_1[1] - t_1[0]
    dt_2 = t_2[1] - t_2[0]
    burn_factor = 7
    
    Mach = np.zeros(np.shape(v))

    for i in range(1, np.size(t_1)):

        v[i] = v[i - 1] + a[i - 1] * dt_1
        d[i] = d[i - 1] + v[i - 1] * dt_1
        m[i] = m[i - 1] - m_dot * dt_1
        drag[i], rho[i] = drag_current(v[i], Cd, d[i], rocket_diameter)
        a[i] = (thrust - m[i] * g - drag[i]) / m[i]
        SOS = SOS_calc(d[i])
        Mach[i] = v[i]/SOS

    for i in range(np.size(t_1), burn_factor * (np.size(t_1)) - 1):

        v[i] = v[i - 1] + a[i - 1] * dt_2
        d[i] = d[i - 1] + v[i - 1] * dt_2
        m[i] = m[i - 1]
        SOS = SOS_calc(d[i])
        Mach[i] = v[i]/SOS

        drag[i], rho[i] = drag_current(v[i], Cd, d[i], rocket_diameter)
        if v[i] >= 0:
            a[i] = (-m[i] * g - drag[i]) / m[i]
        else:
            a[i] = (-m[i] * g + drag[i]) / m[i]
#        
        if d[i] < 0:
            peak_alt = np.max(d)
            return m, d, v, drag, a, rho, peak_alt, Mach

    peak_alt = np.max(d)
    

    return m, d, v, drag, a, rho, peak_alt, Mach


def findThrust(Isp, g, phi, Cd, t_steps, thrust, m_dry, m_dot, desired_alt, rocket_diameter):
    # g, phi, Cd, t_steps, thrust, m_dry, m_dot, desired_alt = variables(Isp, m_dot)
    m_init, m_prop, MR, t_burn = rocket_mass_calc(phi, g, m_dry, thrust, m_dot)
    m, d, v, drag, a, rho = initial_values(phi, g, m_init, t_steps, Cd, rocket_diameter)

    t_1 = np.linspace(0, t_burn, t_steps)

    t_2 = np.linspace(t_burn + t_burn / (t_steps), 5 * t_burn, 4*t_steps)
    t = np.concatenate((t_1, t_2), 0)

    SF = 1.05
    uncert = 0.001
    alt_req = SF * desired_alt

    m, d, v, drag, a, rho, peak_alt, Mach = ballistic_traj(t_1, t_2, v, d, m, drag,
                                                     rho, a, thrust, g, m_dot, Cd, rocket_diameter)

    attempt = 0

    while peak_alt > alt_req + uncert * alt_req or peak_alt < alt_req - uncert * alt_req:

        attempt += 1

        m_dot = m_dot - m_dot * ((peak_alt) / alt_req - 1) * 0.05

        g, t_steps, thrust = variables(
            Isp, m_dot)
        m_init, m_prop, MR, t_burn = rocket_mass_calc(phi, g, m_dry, thrust,
                                                      m_dot)
        t_1 = np.linspace(0, t_burn, t_steps)
        t_2 = np.linspace(t_burn + t_burn / (t_steps), 5 * t_burn, 4*t_steps)
        t = np.concatenate((t_1, t_2), 0)

        m, d, v, drag, a, rho = initial_values(phi, g, m_init, t_steps, Cd, rocket_diameter,t)
        m, d, v, drag, a, rho, peak_alt, Mach = ballistic_traj(
            t_1, t_2, v, d, m, drag, rho, a, thrust, g, m_dot, Cd, rocket_diameter)

    peak_ind = np.where(d == peak_alt)[0]

    d = d[1:peak_ind[0]]
    v = v[1:peak_ind[0]]
    a = a[1:peak_ind[0]]
    drag = drag[1:peak_ind[0]]
    rho = rho[1:peak_ind[0]]
    m = m[1:peak_ind[0]]

    return (thrust, m_dot, m_dry, m_prop, d, t_burn, phi, peak_alt)

def findPhi(Isp, g, phi, Cd, t_steps, thrust, m_dry, m_dot, desired_alt, rocket_diameter,burn_factor):

    m_init, m_prop, MR, t_burn = rocket_mass_calc(phi, g, m_dry, thrust, m_dot)

    
    t_1 = np.linspace(0, t_burn, t_steps)

    t_2 = np.linspace(t_burn + t_burn / (t_steps), burn_factor * t_burn, (burn_factor-1) * t_steps)
    t = np.concatenate((t_1, t_2), 0)
    
    m, d, v, drag, a, rho = initial_values(phi, g, m_init, t_steps, Cd, rocket_diameter,t)

    SF = 1.05
    uncert = 0.001
    alt_req = SF * desired_alt

    m, d, v, drag, a, rho, peak_alt, Mach = ballistic_traj(t_1, t_2, v, d, m, drag,
                                                     rho, a, thrust, g, m_dot, Cd, rocket_diameter)
#    print(peak_alt,alt_req)
    attempt = 0

    while peak_alt > alt_req + uncert * alt_req or peak_alt < alt_req - uncert * alt_req:
        attempt += 1
        
        if phi*m_dry*g > thrust and attempt == 1:
            phi /= phi/2

#        print(peak_alt,alt_req)
        m_dry = m_dry + m_dry * ((peak_alt) / alt_req - 1) * 0.05
#        print(phi, g, m_dry, thrust, m_dot)
        m_init, m_prop, MR, t_burn = rocket_mass_calc(phi, g, m_dry, thrust,
                                                      m_dot)
#        print(m_init, m_prop, MR, t_burn)
        t_1 = np.linspace(0, t_burn, t_steps)
        t_2 = np.linspace(t_burn + t_burn / (t_steps), burn_factor * t_burn, (burn_factor-1) * t_steps)
        t = np.concatenate((t_1, t_2), 0)

        m, d, v, drag, a, rho = initial_values(phi, g, m_init, t_steps, Cd, rocket_diameter,t)
        m, d, v, drag, a, rho, peak_alt, Mach = ballistic_traj(
            t_1, t_2, v, d, m, drag, rho, a, thrust, g, m_dot, Cd, rocket_diameter)

    peak_ind = np.where(d == peak_alt)[0][0]
    ground_ind = np.where(np.diff(np.sign(d)))[0][1]

    d = d[1:ground_ind]
    v = v[1:ground_ind]
    a = a[1:ground_ind]
    drag = drag[1:ground_ind]
    rho = rho[1:ground_ind]
    m = m[1:ground_ind]
    Mach = Mach[1:ground_ind]

    t = t[1:ground_ind]
    t_peak = t[-1]
    
            
    return (thrust, m_dot, m_dry, m_prop, d, t_burn, phi, peak_alt, Mach, t, v)

def findM_dry(Isp, g, phi, Cd, t_steps, thrust, m_dry, m_dot, desired_alt, rocket_diameter,burn_factor, orbital_vel, d_init):

    m_init, m_prop, MR, t_burn = rocket_mass_calc(phi, g, m_dry, thrust, m_dot)

    
    t_1 = np.linspace(0, t_burn, t_steps)

    t_2 = np.linspace(t_burn + t_burn / (t_steps), burn_factor * t_burn, (burn_factor-1) * t_steps)
    t = np.concatenate((t_1, t_2), 0)
    
    m, d, v, drag, a, rho = initial_values(phi, g, m_init, t_steps, Cd, rocket_diameter,t,d_init)

    SF = 1.05
    uncert = 0.001
    alt_req = SF * desired_alt

    m, d, v, drag, a, rho, peak_alt, Mach = ballistic_traj(t_1, t_2, v, d, m, drag,
                                                     rho, a, thrust, g, m_dot, Cd, rocket_diameter)
#    print(peak_alt,alt_req)
    attempt = 0

    while peak_alt > alt_req + uncert * alt_req or peak_alt < alt_req - uncert * alt_req:
        attempt += 1
        
        if phi*m_dry*g > thrust and attempt == 1:
            phi /= phi/2

#        print(peak_alt,alt_req)
            
        m_dry = m_dry + m_dry * ((peak_alt) / alt_req - 1) * 0.05
#        print(phi, g, m_dry, thrust, m_dot)
        m_init, m_prop, MR, t_burn = rocket_mass_calc(phi, g, m_dry, thrust,
                                                      m_dot)
#        print(m_init, m_prop, MR, t_burn)
        t_1 = np.linspace(0, t_burn, t_steps)
        t_2 = np.linspace(t_burn + t_burn / (t_steps), burn_factor * t_burn, (burn_factor-1) * t_steps)
        t = np.concatenate((t_1, t_2), 0)

        m, d, v, drag, a, rho = initial_values(phi, g, m_init, t_steps, Cd, rocket_diameter,t,d_init)
        m, d, v, drag, a, rho, peak_alt, Mach = ballistic_traj(
            t_1, t_2, v, d, m, drag, rho, a, thrust, g, m_dot, Cd, rocket_diameter)

    peak_ind = np.where(d == peak_alt)[0][0]
    ground_ind = np.where(np.diff(np.sign(d)))[0][1]

    d = d[1:ground_ind]
    v = v[1:ground_ind]
    a = a[1:ground_ind]
    drag = drag[1:ground_ind]
    rho = rho[1:ground_ind]
    m = m[1:ground_ind]
    Mach = Mach[1:ground_ind]

    t = t[1:ground_ind]
    t_peak = t[-1]
    
            
    return (thrust, m_dot, m_dry, m_prop, d, t_burn, phi, peak_alt, Mach, t, v, drag, m)


if __name__ == "__main__":  # for testing
#    log_file = ('../log/trajectory.log', 'w')
#    io.config_logger(_LOGGER, log_file, 'w')

    ## Inputs
    g = 9.81
    thrust = 50e3
    Isp = 300
    m_dot = thrust/(Isp*g)
    t_steps = 100
    Cd = 0.3
    m_dry = 100
    rocket_diameter = 1.0
    burn_factor = 7
    desired_alt = 300e3
    
    d_init = 0#10e3
    
    G = 6.673e-11
    M_e = 5.98e24
    R_e = 6.38e6
    
    orbital_vel = np.sqrt(G*M_e/(R_e+desired_alt))
    
    #Inital Guess for phi
    phi = 1.5
    
    thrust, m_dot, m_dry, m_prop, d, t_burn, phi, peak_alt, Mach, t, v, drag, m = findM_dry(Isp, g, phi, Cd, t_steps, thrust, m_dry, m_dot, desired_alt, rocket_diameter, burn_factor, orbital_vel, d_init)
    

#    Calculate total drag from launch to 100km
    index_peak = np.where(d == peak_alt)[0][0]
    index_100km = np.where(np.abs(d[:index_peak]-100e3) == np.min(np.abs(d[:index_peak]-100e3)))[0][0]
    
    delta_v_drag = np.trapz(drag[:index_100km]/m[:index_100km],x = t[:index_100km])
    delta_v_gravity = g*t[index_100km]


    v_exhaust = Isp*g
    extra_flight_time = 100
    delta_v_0 = orbital_vel + 154 + 1422 + extra_flight_time*g
    delta_v_10 = orbital_vel + 38 + 1340 + extra_flight_time*g
    
    m_ratio_0 = np.exp(delta_v_0/v_exhaust)
    m_ratio_10 = np.exp(delta_v_10/v_exhaust)
    m_ratio_0_aero = np.exp(delta_v_0/v_exhaust/1.05)
    

#  
#    m_init, m_prop, MR, t_burn = rocket_mass_calc(phi, g, m_dry, thrust, m_dot)
#    
#    t_1 = np.linspace(0, t_burn, t_steps)
#
#    t_2 = np.linspace(t_burn + t_burn / (t_steps), burn_factor * t_burn, (burn_factor-1) * t_steps)
#    t = np.concatenate((t_1, t_2), 0)
#    
#    m, d, v, drag, a, rho = initial_values(phi, g, m_init, t_steps, Cd, rocket_diameter,t)
#
#    SF = 1.05
#    uncert = 0.001
#    alt_req = SF * desired_alt
#
#    m, d, v, drag, a, rho, peak_alt, Mach = ballistic_traj(t_1, t_2, v, d, m, drag,
#                                                     rho, a, thrust, g, m_dot, Cd, rocket_diameter)
#
#    attempt = 0
#
#    while peak_alt > alt_req + uncert * alt_req or peak_alt < alt_req - uncert * alt_req:
#
#        if phi*m_dry*g > thrust:
#                print('Not enough thrust for initial phi and dry mass!')
#                break
#        attempt += 1
#
#        phi = phi + phi * ((peak_alt) / alt_req - 1) * 0.05
#
#        m_init, m_prop, MR, t_burn = rocket_mass_calc(phi, g, m_dry, thrust,
#                                                      m_dot)
#        t_1 = np.linspace(0, t_burn, t_steps)
#        t_2 = np.linspace(t_burn + t_burn / (t_steps), burn_factor * t_burn, (burn_factor-1) * t_steps)
#        t = np.concatenate((t_1, t_2), 0)
#
#        m, d, v, drag, a, rho = initial_values(phi, g, m_init, t_steps, Cd, rocket_diameter,t)
#        m, d, v, drag, a, rho, peak_alt, Mach = ballistic_traj(
#            t_1, t_2, v, d, m, drag, rho, a, thrust, g, m_dot, Cd, rocket_diameter)
#    
#        peak_ind = np.where(d == peak_alt)[0][0]
#        ground_ind = np.where(np.diff(np.sign(d)))[0][1]
#    
#        d = d[1:ground_ind+1]
#        v = v[1:ground_ind+1]
#        a = a[1:ground_ind+1]
#        drag = drag[1:ground_ind+1]
#        rho = rho[1:ground_ind+1]
#        m = m[1:ground_ind+1]
#        Mach = Mach[1:ground_ind+1]
#    
#        t = t[1:ground_ind+1]
#        t_peak = t[-1]
#        d[-1] = 1
#    
    #    plt.figure(1)
    #    plt.plot(t, d / 100, t, v, t, m)
    #    plt.legend(['Dist/100 (m)', 'Vel (m/s)', 'Mass (kg)'])
    #
    #    plt.figure(2)
    #    plt.plot(t, a, t, drag / 1000, t, rho)
    #    plt.legend(['Acc (m/s2)', 'Drag (kN)', 'Density'])
    print('Burn time =', t_burn)
    print('m_dry =', m_dry)
    print('m_prop =', m_prop)
    print('Altitude =', d[-1])
    print('m_dot =', m_dot)
    print('Thrust =', thrust)
    print('Phi =', phi)
    print('Initial Accel =', (phi-1)*g,'m/s^2')
#
##        plt.figure(10)
##        plt.plot(t[:t_steps+1], d[:t_steps+1]/1000,'r')
##        plt.plot(t[t_steps:], d[t_steps:]/1000,'k')
##        plt.scatter(t[t_steps], d[t_steps]/1000, s=20, c='k')
##        plt.xlabel('Time (s)')
##        plt.ylabel('Altitude (km)')
##        plt.xlim([0,250])
##        plt.savefig('../../Mission_Plan_Final.png',dpi=600)
##    
##        plt.figure(11)
##        plt.plot(t,Mach)
##        plt.xlabel('Time (s)')
##        plt.ylabel('Mach Number')
##        plt.savefig('../../Mach_number.png',dpi=600)
#    
##        plt.figure(13)
##        plt.plot(t,v)
##        plt.xlabel('Time (s)')
##        plt.ylabel('Velocity (m/s)')
##        plt.savefig('../../Velocity_time.png',dpi=600)
#    
#    plt.figure(14)
#    plt.plot(d,v)
#    plt.xlabel('Altitude (m)')
#    plt.ylabel('Velocity (m/s)')
#    plt.xlim([0,15])
#    plt.ylim([0,40])
#    plt.savefig('../../Velocity_alt_launch.png',dpi=600)
    
#        plt.figure(12)
#        plt.plot(d/1000,drag/1000)
#        plt.xlabel('Altitude (km)')
#        plt.ylabel('Drag (kN)')
#        plt.savefig('../../Drag_v_Altitude.png',dpi=600)
