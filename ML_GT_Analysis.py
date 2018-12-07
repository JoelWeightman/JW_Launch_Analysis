# -*- coding: utf-8 -*-
"""
N-Dimensional Machine Learning Genetic Algorithm Adapted for Rocket Trajectory
Joel Weightman

"""


import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.integrate as integ
from multiprocessing import Pool
import Gravity_Turn_Trajectory_ML_shooting as GT
import Gravity_Turn_Trajectory_ML_shooting_bell as GT_bell
import sys
import io

def get_input_variables(input_variables, GT_angle_max):
    
    m_dry = m_dry_max #input_variables[0]
    event_alt = input_variables[0]*event_alt_max + 1e3
    GT_angle = 89*np.pi/180#np.pi/2 - input_variables[2]*GT_angle_max*np.pi/180
    stage_mass_ratios = np.array([0.8,0.2]) 
    accel_init = input_variables[4:6]*1 + 0.5 ## at least 1 gee of accel
#    stage_delta_vee_ratios = np.array([input_variables[3]+input_variables[4],input_variables[4]])/(input_variables[3] + 2*input_variables[4])
    stage_delta_vee_ratios = np.array([input_variables[1],input_variables[2]])*9 + 1 ### MASS RATIOS
    alpha = alpha_max*input_variables[3]
    
    return m_dry, event_alt, GT_angle, stage_mass_ratios, stage_delta_vee_ratios, alpha, accel_init

def calculate_result(engine,pop, weights, m_dry_max, event_alt_max, alpha_max, GT_angle_max, factor, refine):
    
    if refine:
        filename = 'Result_Files/Population_Results_' + engine + '_%d.npy' % filename_num
        pop_ref_all = np.load(filename).item()
        pop_ref = pop_ref_all['actions'][0]
        
        pop['actions'][-1,:] = 0.5

    for i in range(np.shape(pop['actions'])[0]):
        
        if refine:
            input_variables = pop_ref + ((pop['actions'][i,:]-0.5)*factor)*pop_ref
        else:
            input_variables = pop['actions'][i,:]
            
        m_dry, event_alt, GT_angle, stage_mass_ratios, stage_delta_vee_ratios, alpha, accel_init = get_input_variables(input_variables, GT_angle_max)

        if engine == 'bell':
            score, results = GT_bell.run_trajectory(m_dry,stage_mass_ratios,event_alt,GT_angle,stage_delta_vee_ratios,weights,alpha,accel_init)
        else:
            score, results = GT.run_trajectory(m_dry,stage_mass_ratios,event_alt,GT_angle,stage_delta_vee_ratios,weights,alpha,accel_init)
        
        pop['results'][i,:] = results
        pop['score'][i] = score
           
    return pop

def run_last(engine,pop, weights, m_dry_max, event_alt_max, alpha_max, GT_angle_max):
    
    input_variables = pop 
#        m_dry,stage_mass_ratios,stage_m_dots,event_alt,GT_angle,stage_delta_vee_ratios1,stage_delta_vee_ratios2 = [0.2417085,  0.35612504, 0.01455789, 0.6628106,  0.48791092, 0.82595287, 0.31411318]
        
    
    m_dry, event_alt, GT_angle, stage_mass_ratios, stage_delta_vee_ratios, alpha, accel_init = get_input_variables(input_variables, GT_angle_max)

    if engine == 'bell':
        v, phi, r, theta, m, t, ind, grav_delta_vee, score = GT_bell.run_trajectory_final(m_dry,stage_mass_ratios,event_alt,GT_angle,stage_delta_vee_ratios,weights,alpha,accel_init)
    else:
        v, phi, r, theta, m, t, ind, grav_delta_vee, score = GT.run_trajectory_final(m_dry,stage_mass_ratios,event_alt,GT_angle,stage_delta_vee_ratios,weights,alpha,accel_init)

    return v, phi, r, theta, m, t, ind, grav_delta_vee

def init_population(pop):
       
    pop['actions'][:,:] = np.random.uniform(0.00, 0.95, size = np.shape(pop['actions'][:,:]))
   
    return pop

def pop_selection(pop, selection_num):
    
    sorted_pop = np.argsort(pop['score'])[::-1]

    elite_pop = sorted_pop[:selection_num[0]]
    lucky_pop = np.random.choice(sorted_pop[selection_num[1]:],size = selection_num[2], replace = False)
    
    actions = pop['actions'][np.concatenate((elite_pop,lucky_pop))]
    
    selected_pop = sorted_pop[:selection_num[1]]
    mutated_pop = np.random.choice(selected_pop, size = selection_num[3], replace = False)
#    print(selection_num[3])
    selected_pop = np.setdiff1d(selected_pop,mutated_pop)
    
    pop = generate_children(pop, actions, selection_num, selected_pop, mutated_pop)
#    print(pop['actions'][0])
    
    return pop, [np.array(pop['score'][sorted_pop[0]]),pop['results'][sorted_pop[0]],sorted_pop[0]]

def generate_children(pop, actions, selection_num, selected_pop, mutated_pop):
    
    mutated_actions = pop['actions'][mutated_pop,:]
    for i in range(np.size(mutated_actions[:,0])):
        mut_num = np.random.randint(1,selection_num[4]+1,size = 1)[0]
        mut_locs = np.random.randint(0,np.size(mutated_actions[0,:]), size = mut_num)
    
        mutated_actions[i,mut_locs] = np.random.uniform(0, 1, size = mut_num)

 
    random_selection_children_0 = np.random.randint(0,2,size = (int(len(selected_pop)/2),np.size(pop['actions'][0,:])))
    random_selection_children_1 = 1 - random_selection_children_0
#    print(np.shape(random_selection_children_0))
    
    selected_pop_0 = np.random.choice(selected_pop, size = int(len(selected_pop)/2), replace = False)
    selected_pop_1 = np.setdiff1d(selected_pop, selected_pop_0)
    
    children_actions_0 = pop['actions'][selected_pop_0,:]*random_selection_children_0 + pop['actions'][selected_pop_1,:]*random_selection_children_1
    children_actions_1 = pop['actions'][selected_pop_1,:]*random_selection_children_0 + pop['actions'][selected_pop_0,:]*random_selection_children_1
    
    pop['actions'] = np.concatenate((actions,mutated_actions,children_actions_0,children_actions_1), axis = 0)

    return pop

def run(engine, W_vel, W_alt, W_angle, W_mass, perc_elite, perc_lucky, perc_mutation, mutation_chance, pop_size, generations, n_inputs, samples, m_dry_max, event_alt_max, factor, refine, alpha_max, GT_angle_max):
    
#    plt.close("all")
    thresh_perf = 0.01
    
    [W_vel, W_alt, W_angle, W_mass] = np.array([W_vel, W_alt, W_angle, W_mass])/(W_vel + W_alt + W_angle + W_mass)
    
    weights = np.array([W_vel, W_alt, W_angle, W_mass])
   
    ## Conditions to set
    gen_count = np.zeros((2,samples))
    
    for KK in range(samples):
        start_time = time.time()
        
        pop = dict({'actions': np.zeros((pop_size,n_inputs)), 'results':np.zeros((pop_size,4)), 'score':np.zeros((pop_size))})
        
        ## Calculated Parameters
        pop_num_elite = int(perc_elite*pop_size)
        if pop_num_elite < 2:
            pop_num_elite = 2
        pop_num_mutation = int(perc_mutation*pop_size)
        if pop_num_mutation < 1:
            pop_num_mutation = 1
        pop_num_lucky = int((pop_size*perc_lucky))
        if pop_num_lucky < 1:
            pop_num_lucky = 1
        total_non_children = pop_num_lucky + pop_num_elite + pop_num_mutation
        pop_num_selected = pop_size - total_non_children + pop_num_mutation
        if np.mod(pop_num_selected - pop_num_mutation,2) != 0:
            pop_num_selected += 1
            pop_num_elite -= 1
            
        mutation_gene_num = int(mutation_chance*n_inputs)
        if mutation_gene_num < 2:
            mutation_gene_num = 2
        selection_num = np.array([pop_num_elite, pop_num_selected, pop_num_lucky, pop_num_mutation, mutation_gene_num])
            
        theory_max = np.sum(weights)
        
        COUNT = 0
        ## First iteration
        pop = init_population(pop)
        old_best = 0
        count = 0
        ## Calculate Generations until convergence or theory max               
        for I in range(generations):
            
            pop = calculate_result(engine,pop, weights, m_dry_max, event_alt_max, alpha_max, GT_angle_max, factor,refine)
            pop, best_performance = pop_selection(pop, selection_num)
            
#            if best_performance[0] < 0:
#                pop = init_population(pop)
            print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (n_inputs,pop_size,KK, theory_max))
            print('Gen %1d Performance %3.3f, Velocity = %3.1fms, Altitude = %3.3fm, Angle = %3.3fdeg, Mass = %3.3fkg' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2]*180/np.pi, best_performance[1][3]))
#            if I == generations - 1:
#                print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (n_inputs,pop_size,KK, theory_max))
#                print('Gen %1d Performance %3.3f, Velocity = %3.1f, Altitude = %3.3f, Angle = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2]*180/np.pi))
#                
#            if best_performance[0] >= theory_max-thresh_perf:
#                COUNT += 1
#                if COUNT >= 2:
#                    print('Gen %1d Performance %3.3f, Velocity = %3.1f, Altitude = %3.3f, Angle = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2]*180/np.pi))
#                    break
            if best_performance[0] == old_best and best_performance[1][2]*180/np.pi < 1.0 and refine == True:
                count += 1
                if count == 10:
                    break
            else:
                count = 0
            old_best = best_performance[0]
        
#        print(A_l, A_v, A_t)
        elapsed_time = time.time() - start_time
        gen_count[0,KK] = I
        gen_count[1,KK] = elapsed_time

    gen_count_stats = np.zeros((3))
    gen_count_stats[:2] = np.mean(gen_count,1)
    gen_count_stats[2] = np.sqrt(np.var(gen_count[0,:]))
    
    return gen_count_stats, pop, best_performance

def n_d_runfile(engine,W_vel, W_alt, W_angle, W_mass, perc_elite = 1, perc_lucky = 1, perc_mutation = 1, perc_selected = 1, mutation_chance = 1, samples = 1, generations = 1, m_dry_max = 1, event_alt_max = 1, factor = 1, refine = False, alpha_max = 0, GT_angle_max = 1):

    perc_elite, perc_lucky, perc_mutation, perc_selected = np.array([perc_elite, perc_lucky, perc_mutation, perc_selected])/(perc_elite + perc_lucky + perc_mutation + perc_selected)

    gen_count_stats, pop, best_performance = run(engine,W_vel, W_alt, W_angle, W_mass, perc_elite, perc_lucky, perc_mutation, mutation_chance, pop_size, generations, n_inputs, samples, m_dry_max, event_alt_max, factor, refine, alpha_max, GT_angle_max)
        
    if refine == False:
        filename = 'Result_Files/Population_Results_' + engine + '_%d.npy' % filename_num
        np.save(filename, pop)
    else:
                
        filename = 'Result_Files/Population_Results_' + engine + '_%d.npy' % filename_num
        pop_orig_all = np.load(filename).item()
                
        pop['actions'] = (pop['actions']-0.5)*factor*pop_orig_all['actions'][0] + pop_orig_all['actions'][0]
        
        filename = 'Result_Files/Population_Results_' + engine + '_%d.npy' % filename_num
        np.save(filename, pop)
    
    v, phi, r, theta, m, t, ind, grav_delta_vee = run_last(engine,pop['actions'][0], np.array([0,0,0,0]), m_dry_max, event_alt_max, alpha_max, GT_angle_max)
    
    return gen_count_stats, pop, v, phi, r, theta, m, t, ind, grav_delta_vee

if __name__ == "__main__":
       
    plt.close('all')

    already_run = False
    ready_for_refine = False
    ready_for_refine2 = True
    
    global filename_num
    filename_num = 3
    
    n_inputs = 6
    factor = 1
    engine = 'bell'
    
    pop_size = 100
    W_vel = 0.5
    W_alt = 1.0
    W_angle = 1.0
    W_mass = 0.25
    
    ## Other Set Params
    perc_elite = 0.4
    perc_lucky = 0.05
    perc_mutation = 0.8
    perc_selected = 0.1
    mutation_chance = 0.5
    
    generations = 1000
    samples = 1
    m_dry_max = 0.15e3
    event_alt_max = 5e3
    alpha_max = -5 ## In degrees
    alpha_max*= np.pi/180
    GT_angle_max = 2
    
    if ready_for_refine == False:
        if already_run == False:
            start_time = time.time()
            generation_stats, pop, v, phi, r, theta, m, t, ind, grav_delta_vee = n_d_runfile(engine,W_vel, W_alt, W_angle, W_mass, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, samples, generations, m_dry_max, event_alt_max, factor, False, alpha_max, GT_angle_max)
            print(time.time()-start_time)
            
            pop_best = pop['actions'][0]
            ready_for_refine = True
            
        else:
            filename = 'Result_Files/Population_Results_' + engine + '_%d.npy' % filename_num
            pop_ref_all = np.load(filename).item()
            pop_best = pop_ref_all['actions'][0]
            ready_for_refine = True
##    
    if ready_for_refine == True:
        factor = 0.1
        if already_run == False:
            generation_stats, pop, v, phi, r, theta, m, t, ind, grav_delta_vee = n_d_runfile(engine,W_vel, W_alt, W_angle, W_mass, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, samples, generations, m_dry_max, event_alt_max, factor, True, alpha_max, GT_angle_max)
            pop_best = pop['actions'][0]
            ready_for_refine2 = True
            
        else:
            filename = 'Result_Files/Population_Results_' + engine + '_%d.npy' % filename_num
            pop_ref_all = np.load(filename).item()
            pop_best = pop_ref_all['actions'][0]
            ready_for_refine2 = True
            

    if ready_for_refine2 == True:
        factor = 0.01
        if already_run == False:
            generation_stats, pop, v, phi, r, theta, m, t, ind, grav_delta_vee = n_d_runfile(engine,W_vel, W_alt, W_angle, W_mass, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, samples, generations, m_dry_max, event_alt_max, factor, True, alpha_max, GT_angle_max)
            pop_best = pop['actions'][0]

        else:
            filename = 'Result_Files/Population_Results_' + engine + '_%d.npy' % filename_num
            pop_ref_all = np.load(filename).item()
            pop_best = pop_ref_all['actions'][0]
            
    input_variables = pop_best

    m_dry, event_alt, GT_angle, stage_mass_ratios, stage_delta_vee_ratios, alpha, accel_init = get_input_variables(input_variables, GT_angle_max)
    
    v, phi, r, theta, m, t, ind, delta_vee_grav = run_last(engine,pop_best, np.array([0,0,0,0]), m_dry_max, event_alt_max, alpha_max, GT_angle_max)
    ind_burn = ind + 2000 - 4
    
    g = 9.80665
    R_e = 6.38e6
    m_dot_1 = -(m[1]-m[0])/(t[1]-t[0])
    m_dot_2 = -(m[2000] - m[1999])/(t[2000]-t[1999])
    thrust_SL = m[0]*(1+accel_init[0])*g
    Isp_SL = thrust_SL/(m_dot_1*g)
    
    D = GT.drag_current(v,0.2,r-R_e,0.5)
    delta_vee_drag = np.sum(D[:2000]/m[:2000]*(t[1:2001] - t[:2000]))
    
    