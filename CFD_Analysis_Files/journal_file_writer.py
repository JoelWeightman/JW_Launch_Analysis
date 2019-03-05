# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:44:34 2019

@author: JL
"""

import numpy as np

# generate list of pressures to be used for the cases
input_pressure = 500000 #Pa
ratios = np.array([0.5,0.75,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.25,2.5,2.75,3.0,3.5,4.0,4.5,5.0,6,7,8,9,10])
pressure_list = input_pressure/ratios

### Write journal file
file_obj = open('D:\Rocket\Cold_Flow_Nozzle\CF_TVC_Truncation_PR50_Sweep_60_files\dp13\FLU\Fluent\Journal_File.txt','w')

#not sure if this does anything :/
file_obj.write("/file/confirm-overwrite no\n")

#Load case file previously made
file_obj.write(";/file/read-case-data D:/Rocket/Cold_Flow_Nozzle/CF_TVC_Truncation_PR50_Sweep_60_files/dp13/FLU/Fluent/Sweep_File.cas\n")
file_obj.write(';\n')

for i,pressure_current in enumerate(pressure_list):
    file_out1 = '"outputs/output_Inlet_2_Pressure_' + str(int(pressure_list[i])) + '.out"'
    file_out2 = '"outputs/output_Inlet_2_Pressure_' + str(int(pressure_list[i])) + '.dat"'
    
    # Change boundary condition (replace outlet(x2) with inlet for inlet boundary) yes's and no's are for the other settings (300K for Temp, 5 and 10 for turbulence settings)
    set_pressure = "/define/boundary-conditions/pressure-inlet inlet_2 yes no " + str(pressure_current) + " no " + str(pressure_current) + " no 300 no yes no no yes 5 10\n"
    
    # set the output filename for the output parameters. This will write a separate file for each case containing the output params.
    output_file_param = '/define/parameters/output-parameters/write-all-to-file ' + file_out1 + '\n'
    output_file_data = '/file/write-case-data ' + file_out2 + '\n'
    
    file_obj.write(set_pressure)
    file_obj.write("/solve/initialize/hyb-initialization\n")  # Initialise using hybrid (with settings selected when case is loaded)
    file_obj.write("/solve/iterate 30000 y\n") # iterate X times. The "y" selects yes to overwrite datafile.
    file_obj.write(output_file_param) # Output params
    file_obj.write(output_file_data) # Output data file for CFD post
    file_obj.write(';\n')

file_obj.close()