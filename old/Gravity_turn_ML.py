# -*- coding: utf-8 -*-
"""
N-Dimensional Machine Learning Genetic Algorithm
Joel Weightman

A particle is accelerated at time t_0 at a value of acc_0, at t_1, the acceleration 
changes to acc_1. Cumulative sum is used for integration. No smoothing.

"""


import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.integrate as integ
from multiprocessing import Pool
import Gravity_Turn_Trajectory_ML

def init_population(pop, dimensions, deg_steps):
       
    if dimensions == 1:
        pop['actions'][:,:,0] = np.random.randint(-1, 2, size = np.shape(pop['actions'][:,:,0]))
    elif dimensions == 2:
        pop['actions'][:,:,0] = np.random.randint(0, 2, size = np.shape(pop['actions'][:,:,0]))
#        pop['actions'][:,:,1] = np.random.uniform(0, 2*np.pi, size = np.shape(pop['actions'][:,:,1]))
        pop['actions'][:,:,1] = np.random.randint(0, 360/deg_steps, size = np.shape(pop['actions'][:,:,1]))*deg_steps*(np.pi/180)
#    print(pop['actions'][:,:,1])
    return pop

def pop_selection(pop, selection_num, dimensions, angle_mutation, deg_steps):
    
    sorted_pop = np.argsort(pop['score'])[::-1]
    
    elite_pop = sorted_pop[:selection_num[0]]
    selected_pop = sorted_pop[:selection_num[1]]
    lucky_pop = np.random.choice(sorted_pop[selection_num[1]:],size = selection_num[2], replace = False)
    mutated_pop = np.random.choice(selected_pop, size = selection_num[3], replace = False)
#    print(selection_num[3])
    selected_pop = np.setdiff1d(selected_pop,mutated_pop)
    
    actions = pop['actions'][np.concatenate((elite_pop,lucky_pop))]
    
    pop = generate_children(pop, actions, selection_num, selected_pop, mutated_pop, dimensions, angle_mutation, deg_steps)
    
    return pop, [np.array(pop['score'][sorted_pop[0]]),pop['results'][sorted_pop[0]],sorted_pop[0]]

def generate_children(pop, actions, selection_num, selected_pop, mutated_pop, dimensions, angle_mutation, deg_steps):
    
    mutated_actions = pop['actions'][mutated_pop,:,:]
    mut_num = np.random.randint(1,selection_num[4]+1,size = 1)[0]
    indices = np.random.randint(0,np.size(mutated_actions[0,:,0]), size = (np.size(mutated_actions[:,0,0]),mut_num))
    for i, mut_locs in enumerate(indices):
        if dimensions == 1:
            mutated_actions[i,mut_locs,0] = np.random.randint(-1, 2, size = mut_num)
        elif dimensions == 2:
            mutated_actions[i,mut_locs,0] = np.random.randint(0, 2, size = mut_num)
    if dimensions == 2:
        mut_num = np.random.randint(1,selection_num[4]+1,size = 1)[0]
        indices = np.random.randint(0,np.size(mutated_actions[0,:,0]), size = (np.size(mutated_actions[:,0,0]),mut_num))
        for i, mut_locs in enumerate(indices):
#            mutated_actions[i,mut_locs,1] = np.random.uniform(0, 2*np.pi, size = mut_num)
            mutated_actions[i,mut_locs,1] = np.random.randint(0, 360/deg_steps, size = mut_num)*deg_steps*(np.pi/180)
#            mutated_actions[i,mut_locs,1] = mutated_actions[i,mut_locs,1]*np.random.uniform(1-angle_mutation, 1+angle_mutation, size = mut_num)
#            mutated_actions[i,mut_locs,1] = np.floor((mutated_actions[i,mut_locs,1]*np.random.uniform(1-angle_mutation, 1+angle_mutation, size = mut_num))*180/np.pi)*np.pi/180
            
    random_selection_children_0 = np.random.randint(0,2,size = (int(len(selected_pop)/2),np.size(pop['actions'][0,:,0]),dimensions))
    if dimensions == 2:
        random_selection_children_0[:,:,1] = random_selection_children_0[:,:,0]
    random_selection_children_1 = 1 - random_selection_children_0
#    print(np.shape(random_selection_children_0))
    
    selected_pop_0 = np.random.choice(selected_pop, size = int(len(selected_pop)/2), replace = False)
    selected_pop_1 = np.setdiff1d(selected_pop, selected_pop_0)
    
    children_actions_0 = pop['actions'][selected_pop_0,:,:]*random_selection_children_0 + pop['actions'][selected_pop_1,:,:]*random_selection_children_1
    children_actions_1 = pop['actions'][selected_pop_1,:,:]*random_selection_children_0 + pop['actions'][selected_pop_0,:,:]*random_selection_children_1
    
    pop['actions'] = np.concatenate((actions,mutated_actions,children_actions_0,children_actions_1), axis = 0)

    return pop

def run(W_vel, W_alt, W_angle, perc_elite, perc_lucky, perc_mutation, mutation_chance, pop_size, generations, t_steps, samples):
    
#    plt.close("all")
    
    # Threshold Values
    thresh_loc = 0.0
    thresh_vel = 0.0
    thresh_time = 0.0
    thresh_perf = 1e-5
    
    thresh = np.array([thresh_loc, thresh_vel, thresh_time])
    weights = np.array([W_vel, W_alt, W_angle])
   
    ## Conditions to set
    gen_count = np.zeros((2,samples))
    
    for KK in range(samples):
        start_time = time.time()
        
        pop = dict({'actions': np.zeros((pop_size,t_steps,dimensions)), 'results':np.zeros((pop_size,dimensions + 1)), 'score':np.zeros((pop_size))})
        
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
            
        mutation_gene_num = int(mutation_chance*t_steps)
        if mutation_gene_num < 2:
            mutation_gene_num = 2
        selection_num = np.array([pop_num_elite, pop_num_selected, pop_num_lucky, pop_num_mutation, mutation_gene_num])
            
        theory_max = (W_vel + W_alt + W_angle)
        
        COUNT = 0
        ## First iteration
        pop = init_population(pop)
        ## Calculate Generations until convergence or theory max               
        for I in range(generations):
            
            pop = calculate_result(pop, acc, t_steps, dt, [], False, dimensions)
            pop = fitness(pop, weights, goals, thresh, v_max, dimensions)
            pop, best_performance = pop_selection(pop, selection_num, dimensions, angle_mutation, deg_steps)
            
            if np.mod(I,100) == 0 and I != 0:
                print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (t_steps,pop_size,KK, theory_max))
                if dimensions == 1:
                    print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                elif dimensions == 2:
                    print('Gen %1d Performance %3.3f, Distance_x = %3.1f, Distance_y = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2]))
#                    for III in range(t_steps):
#                        print(pop['actions'][0,III,1]*180/(np.pi))
#                calculate_result(pop, acc, t_steps, dt, 0, True, dimensions)
            if I == generations - 1:
                print('t: %d, Pop: %d, Run: %1d, Max Perf: %3.3f' % (t_steps,pop_size,KK, theory_max))
                if dimensions == 1:
                    print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                elif dimensions == 2:
                    print('Gen %1d Performance %3.3f, Distance_x = %3.1f, Distance_y = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2]))
#                calculate_result(pop, acc, t_steps, dt, 0, True, dimensions)
                
            if best_performance[0] >= theory_max-thresh_perf:
                COUNT += 1
                if COUNT >= 2:
                    if dimensions == 1:
                        print('Gen %1d Performance %3.3f, Distance = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1]))
                    elif dimensions == 2:
                        print('Gen %1d Performance %3.3f, Distance_x = %3.1f, Distance_y = %3.1f, Velocity = %3.3f' % (I, best_performance[0], best_performance[1][0], best_performance[1][1], best_performance[1][2]))
                        
                    break
        
#        print(A_l, A_v, A_t)
        elapsed_time = time.time() - start_time
        gen_count[0,KK] = I
        gen_count[1,KK] = elapsed_time

    gen_count_stats = np.zeros((3))
    gen_count_stats[:2] = np.mean(gen_count,1)
    gen_count_stats[2] = np.sqrt(np.var(gen_count[0,:]))
    
    return gen_count_stats, pop, best_performance, acc, dt

def n_d_runfile(dimensions = 1, loc_des_x = 1, loc_des_y = 1, vel_des = 1, t_steps = 1, pop_size = 1, A_l = 1, A_v = 1, A_f = 1, A_t = 1, perc_elite = 1, perc_lucky = 1, perc_mutation = 1, perc_selected = 1, mutation_chance = 1, angle_mutation = 1, samples = 1, generations = 1):

    dt = 1.0
    ideal_time_perc = 0.8
    perc_elite, perc_lucky, perc_mutation, perc_selected = np.array([perc_elite, perc_lucky, perc_mutation, perc_selected])/(perc_elite + perc_lucky + perc_mutation + perc_selected)

    gen_count_stats, pop, best_performance, acc, dt = run(A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, mutation_chance, pop_size, generations, t_steps, loc_des_x, loc_des_y, vel_des, ideal_time_perc, dt, dimensions, samples, angle_mutation)
    
    calculate_result(pop, acc, t_steps, dt, best_performance[2], True, dimensions)
    print(gen_count_stats)
    print(pop['actions'][0,:,0])
    print(pop['actions'][0,:,1]*180/(np.pi))
    return gen_count_stats


def fitness_ML(pop_ML):
    
    pop_ML['score'] = np.exp(-(pop_ML['results']/100))
    
    return pop_ML

def init_population_ML(pop_ML):
    
    pop_ML['actions'] = np.random.uniform(0.025,0.975,size = (np.shape(pop_ML['actions'])[0],np.shape(pop_ML['actions'])[1]))
    
    return pop_ML

def calculate_ML(pop_ML, current_gen, ML_settings):
    
    start_time = time.time()
    gen_stats = np.zeros((np.size(pop_ML['actions'],0)))
    pool = Pool(10)
    
    ML_dimensions, ML_loc_des_x, ML_loc_des_y, ML_vel_des, ML_t_steps, ML_pop_size_max, ML_samples, ML_generations = ML_settings
  
    A = []
       
    for i in range(np.size(pop_ML['actions'],0)):
        A.append((ML_dimensions,ML_loc_des_x,ML_loc_des_y,ML_vel_des,ML_t_steps,int(pop_ML['actions'][i,0]*ML_pop_size_max),pop_ML['actions'][i,1],pop_ML['actions'][i,2],pop_ML['actions'][i,3],pop_ML['actions'][i,4],pop_ML['actions'][i,5],pop_ML['actions'][i,6],pop_ML['actions'][i,7],pop_ML['actions'][i,8],pop_ML['actions'][i,9],pop_ML['actions'][i,10],ML_samples,ML_generations))
#    print(A)
    print('Gen = %d' %(current_gen))
    gen_stats = np.zeros((np.shape(A)[0]))
#    for i in range(np.shape(A)[0]):
#        dimensions, loc_des_x, loc_des_y, vel_des, t_steps, pop_size, A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, angle_mutation, samples, generations = A[i]
#        gen_stats[i] = n_d_runfile(dimensions, loc_des_x, loc_des_y, vel_des, t_steps, pop_size, A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, angle_mutation, samples, generations)
#    print(resultant_gen)
    for i, result in enumerate(pool.starmap(n_d_runfile, A)):
        print(current_gen, i, result)
        gen_stats[i] = result[0]
        
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    
    pop_ML['results'] = gen_stats
    pool.close()
    print(np.median(gen_stats))
    
    return pop_ML

def pop_selection_ML(pop_ML, selection_num_ML):
    
    sorted_pop = np.argsort(pop_ML['score'])[::-1]
    
    elite_pop = sorted_pop[:selection_num_ML[0]]
    selected_pop = sorted_pop[:selection_num_ML[1]]
    if len(selected_pop) == np.shape(pop_ML['actions'])[0]:
        lucky_pop = []
    else:
        lucky_pop = np.random.choice(sorted_pop[selection_num_ML[1]:],size = selection_num_ML[2], replace = False)
    mutated_pop = np.random.choice(selected_pop, size = selection_num_ML[3], replace = False)
    selected_pop = np.setdiff1d(selected_pop,mutated_pop)
    lucky_elite = np.concatenate((elite_pop,lucky_pop))
    if len(lucky_elite) == 0:
        actions = []
    else:
        actions = pop_ML['actions'][lucky_elite]
    
    pop_ML = generate_children_ML(pop_ML, actions, selection_num_ML, selected_pop, mutated_pop)
    
    return pop_ML, [np.array(pop_ML['score'][sorted_pop[0]]),pop_ML['results'][sorted_pop[0]],sorted_pop[0]]

def generate_children_ML(pop_ML, actions, selection_num_ML, selected_pop, mutated_pop):
    
    mutated_actions = pop_ML['actions'][mutated_pop,:]
    mut_num = np.random.randint(1,selection_num_ML[4]+1,size = 1)[0]
    indices = np.random.randint(0,np.size(mutated_actions[0,:]), size = (np.size(mutated_actions[:,0]),mut_num))
    for i, mut_locs in enumerate(indices):
        mutated_actions[i,mut_locs] = np.random.uniform(0.025, 0.975, size = mut_num)
    
    random_selection_children_0 = np.random.randint(0,2,size = (int(len(selected_pop)/2),np.size(pop_ML['actions'][0,:])))
    random_selection_children_1 = 1 - random_selection_children_0
    
    selected_pop_0 = np.random.choice(selected_pop, size = int(len(selected_pop)/2), replace = False)
    selected_pop_1 = np.setdiff1d(selected_pop, selected_pop_0)
    
    children_actions_0 = pop_ML['actions'][selected_pop_0,:]*random_selection_children_0 + pop_ML['actions'][selected_pop_1,:]*random_selection_children_1
    children_actions_1 = pop_ML['actions'][selected_pop_1,:]*random_selection_children_0 + pop_ML['actions'][selected_pop_0,:]*random_selection_children_1

    pop_ML['actions'] = np.concatenate((actions,mutated_actions,children_actions_0,children_actions_1), axis = 0)

    return pop_ML

def run_ML(generations, ML_settings, already_started = 0, pop_ML = None):

    pop_size = 50
    t_steps = 11
    
    perc_elite = 0.10
    perc_lucky = 0.05
    perc_mutation = 0.20
#    mutation_chance = 0.05
    
    pop_num_elite = int(perc_elite*pop_size)
    pop_num_mutation = int(perc_mutation*pop_size)
    pop_num_lucky = int((pop_size*perc_lucky))
    total_non_children = pop_num_lucky + pop_num_elite + pop_num_mutation
    pop_num_selected = pop_size - total_non_children + pop_num_mutation
    
    if np.mod(pop_num_selected - pop_num_mutation,2) != 0:
        pop_num_elite += 1
        pop_num_selected -= 1
            
    mutation_gene_num = 1#int(mutation_chance*t_steps)
    selection_num_ML = np.array([pop_num_elite, pop_num_selected, pop_num_lucky, pop_num_mutation, mutation_gene_num])
    
    time_to_locvel_ratio = 0.99
    dist_to_vel_ration = 0.5
    
    if already_started == 0:
        print('Starting')
        pop_ML = dict({'actions': np.zeros((pop_size,t_steps)), 'results':np.zeros((pop_size)), 'score':np.zeros((pop_size))})
        pop_ML = init_population_ML(pop_ML)      
    else:
        print('Already Started, but Continuing')
    
    for I in range(generations):
              
        vel_loc_weight = (pop_ML['actions'][:,1] + pop_ML['actions'][:,2])
        bad_time_weights = pop_ML['actions'][:,4] > vel_loc_weight*time_to_locvel_ratio
        pop_ML['actions'][bad_time_weights,4] = (pop_ML['actions'][bad_time_weights,1] + pop_ML['actions'][bad_time_weights,2])*time_to_locvel_ratio

        pop_ML = calculate_ML(pop_ML, I, ML_settings)
        pop_ML = fitness_ML(pop_ML)
        pop_ML, best_performance = pop_selection_ML(pop_ML, selection_num_ML)
        
        print('BEST PERFORMANCE AT GEN %d: %3.3f' %(I, best_performance[0]))
        
        filename = 'Population_ML_Gen_Temp.npy'
        np.save(filename, pop_ML)
       
    return pop_ML, best_performance

def start_ML_ML_optimization(generations):
           
    change_settings = 1
    already_started = 1
    load_old_file = 0
    
    ML_dimensions = 2
    ML_loc_des_x = 100
    ML_loc_des_y = 100*np.tan(60*np.pi/180)
    ML_vel_des = 0
    ML_generations = 1000
    ML_t_steps = 5
    ML_samples = 5
    ML_pop_size_max = 5000
    
    ML_settings = [ML_dimensions, ML_loc_des_x, ML_loc_des_y, ML_vel_des, ML_t_steps, ML_pop_size_max, ML_samples, ML_generations]
    
    if already_started == 0:
        pop_ML, best_performance = run_ML(generations, ML_settings, already_started, [])
    else:
        if load_old_file == 0:
            file_base = str(ML_dimensions) + 'D_' + str(generations) + '_Gens_time_ML_t_' + str(ML_t_steps) + '_samp_' + str(ML_samples) + '_pop_' + str(ML_pop_size_max)
            filename = file_base + '_settings_1.npy'
            pop_ML = np.load(filename).item()
            filename = file_base + '_settings_2.npy'
            ML_settings = np.load(filename).astype(int)
            ML_dimensions, ML_loc_des_x, ML_loc_des_y, ML_vel_des, ML_t_steps, ML_pop_size_max, ML_samples, ML_generations = ML_settings
            ML_settings_temp = np.load(filename) 
            ML_loc_des_x, ML_loc_des_y, ML_vel_des = ML_settings_temp[1:4]
    #        filename = 'Population_ML_Gen_Temp.npy'
    #        pop_ML = np.load(filename).item()
            if change_settings == 1:
                ML_generations = 2000
                ML_t_steps = 10
#                ML_samples = 5
#                ML_dimensions = 2
#                ML_loc_des_y = 100*np.tan(60*np.pi/180)
                generations = 10
#                ML_pop_size_max = 5000
        else:
            file_base = '1D_200_Gens_time_ML_t_50_samp_20_pop_2000'
            filename = file_base + '_settings_1.npy'
            pop_ML = np.load(filename).item()
            
        
        ML_settings = [ML_dimensions, ML_loc_des_x, ML_loc_des_y, ML_vel_des, ML_t_steps, ML_pop_size_max, ML_samples, ML_generations]
        
        pop_ML, best_performance = run_ML(generations, ML_settings, already_started, pop_ML)
    
    filename = str(ML_dimensions) + 'D_' + str(generations) + '_Gens_time_ML_t_' + str(ML_t_steps) + '_samp_' + str(ML_samples) + '_pop_' + str(ML_pop_size_max) + '_settings_1.npy'
    np.save(filename, pop_ML)
    filename = str(ML_dimensions) + 'D_' + str(generations) + '_Gens_time_ML_t_' + str(ML_t_steps) + '_samp_' + str(ML_samples) + '_pop_' + str(ML_pop_size_max) + '_settings_2.npy'
    np.save(filename, ML_settings)

    return

if __name__ == "__main__":
       
    plt.close('all')
    ### Design Values
    
    optimize_ML = 0
    generations = 20
    already_started = 1    
    
    if optimize_ML == 0:
    
        if already_started == 1:
            ML_t_steps = 5
            ML_samples = 5
            ML_pop_size_max = 5000
            ML_dimensions = 2
            
            file_base = str(ML_dimensions) + 'D_' + str(generations) + '_Gens_time_ML_t_' + str(ML_t_steps) + '_samp_' + str(ML_samples) + '_pop_' + str(ML_pop_size_max)
#            file_base = '1D_100_Gens_ML_t_50_samp_10_pop_1000'
            
            filename = file_base + '_settings_1.npy'
            pop_ML = np.load(filename).item()
            pop_size, A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, angle_mutation = pop_ML['actions'][0,0],pop_ML['actions'][0,1],pop_ML['actions'][0,2],pop_ML['actions'][0,3],pop_ML['actions'][0,4],pop_ML['actions'][0,5],pop_ML['actions'][0,6],pop_ML['actions'][0,7],pop_ML['actions'][0,8],pop_ML['actions'][0,9],pop_ML['actions'][0,10]
    
            filename = file_base + '_settings_2.npy'
            ML_settings = np.load(filename).astype(int)
            dimensions, loc_des_x, loc_des_y, vel_des, t_steps, pop_size_max, samples, generations = ML_settings
            ML_settings_temp = np.load(filename) 
            loc_des_x, loc_des_y, vel_des = ML_settings_temp[1:4]
            pop_size = int(pop_size*pop_size_max)
            
            
            t_steps = 10
#            samples = 1
#            loc_des_y = 100*np.tan(60*np.pi/180)
#            angle_mutation = 0.05
#            dimensions = 2
#            generations = 10000
#            pop_size = 2000
#            A_l = 0.5
#            A_v = 0.5
#            A_t = (A_v+A_t)*0.99
            
#            A_f = 2
#            A_f, A_l, A_t, A_v = [0.5,0.01,0.426522442525838,0.7690407264472648]
#            perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance = [0.18,0.12,0.1,0.82,0.13]
            
        
        else:
            dimensions = 2
            loc_des_x = 100
            loc_des_y = 100*np.tan(60*np.pi/180)
            vel_des = 0  
            t_steps = 5
            
            pop_size = 100
            A_l = 0.3
            A_v = 0.6
            A_f = 0
            A_t = 0.2
            
            ## Other Set Params
            perc_elite = 0.2
            perc_lucky = 0.15
            perc_mutation = 0.2
            perc_selected = 0.45
            mutation_chance = 0.4
            angle_mutation = 0.1
            
            generations = 1000
            samples = 1
        
        generation_stats = n_d_runfile(dimensions, loc_des_x, loc_des_y, vel_des, t_steps, pop_size, A_l, A_v, A_f, A_t, perc_elite, perc_lucky, perc_mutation, perc_selected, mutation_chance, angle_mutation, samples, generations)
    
#        plt.figure(1000)
#        plt.plot(pop_size,generation_stats[0],'.k')
#        
#        plt.figure(1002)
#        plt.plot(pop_size,generation_stats[2],'.r')
#        
#        plt.figure(1001)
#        plt.plot(pop_size,generation_stats[1],'.b')
    
    elif optimize_ML == 1:
        start_ML_ML_optimization(generations)
        
        
        # Optimized 20 t_step for
#        time [3.30000000e+01 2.30053663e-02 3.65106365e-02]
#        generations [2.90000000e+01 5.80129623e-02 1.33688030e-02]
#    filename = 'Population_ML_Gen_Temp.npy'
#    pop_ML = np.load(filename).item()