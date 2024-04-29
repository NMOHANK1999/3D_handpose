# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:49:30 2023

@author: NIshanth Mohankumar
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def out(tim):
    data = np.loadtxt("kpts_3d.dat", dtype='float')
    time_per_iteration = tim / len(data)
    # Extract x, y, z coordinates for thumb, index, and base
    x_data = data[:, [0, 4, 8]]
    y_data = data[:, [1, 5, 9]]  
    z_data = data[:, [2, 6, 10]] 
    
    for i in range(1, len(x_data)):
        if x_data[i, 0] == -1:
            x_data[i, :] = x_data[i-1, :]
            y_data[i, :] = y_data[i-1, :]
            z_data[i, :] = z_data[i-1, :]
            
            
    x_data = gaussian_filter1d(x_data, sigma=1, axis=0, mode='constant')
    y_data = gaussian_filter1d(y_data, sigma=1, axis=0, mode='constant')
    z_data = gaussian_filter1d(z_data, sigma=1, axis=0, mode='constant')
    

if __name__ == "__main__":
    data = np.loadtxt("kpts_3d.dat", dtype='float')
    #4 is thumb and 8 is index
    
    #can use camera frame time for this but for now lets leave it like this
    FPS = 60
    time = 1/FPS

    # Extract x, y, z coordinates for thumb, index, and base
    x_data = data[:, [0, 4, 8]]
    y_data = data[:, [1, 5, 9]]  
    z_data = data[:, [2, 6, 10]] 
    
    for i in range(1, len(x_data)):
        if x_data[i, 0] == -1:
            x_data[i, :] = x_data[i-1, :]
            y_data[i, :] = y_data[i-1, :]
            z_data[i, :] = z_data[i-1, :]
            
            
    x_data = gaussian_filter1d(x_data, sigma=1, axis=0, mode='constant')
    y_data = gaussian_filter1d(y_data, sigma=1, axis=0, mode='constant')
    z_data = gaussian_filter1d(z_data, sigma=1, axis=0, mode='constant')
    
    data = np.stack([x_data, y_data, z_data], axis = 2)
    
    disp_along_axis = []
    vel_along_axis = []
    angle = []
    vel = []
    disp = []
    
    for i in range(1, len(x_data)):
        disp_along_axis.append((data[i, :, :] - data[i-1, :, :])) 
    
    #distance and displacement i believe is in pixels
    disp_along_axis = np.array(disp_along_axis)
    disp = np.linalg.norm(disp_along_axis, axis = 2)
    disp_base = disp[:, 0]
    
    vel_along_axis = disp_along_axis/time
    vel = np.linalg.norm(vel_along_axis, axis = 2)
    vel_base = vel[:, 0]
    
    #angles
    #vec_base_to_thumb 
    a = data[:, 0 , :] - data[:, 1, :]
    #vec_base_to_index 
    b = data[:, 0 , :] - data[:, 2, :]
    #vec_thumb_to_index 
    c = data[:, 1 , :] - data[:, 2, :]
    #angle = np.cos angle btw the 2 and sorted
    
    angle = np.argcos((a**2 + b**2 - c**2)/(2*a*b)) * 180 / 3.14  
    
        
    
    
    

            
    
    
        