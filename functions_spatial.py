#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from tqdm import tqdm

import seaborn as sns

import matplotlib as mpl
mpl.rcParams['font.size']=12

# Initial circle population
def fill_circle(grid, a):
    grid_0 = np.copy(grid)
    radius = int(a*grid.shape[0])
    iC = int(grid.shape[0]/2)
    jC = int(grid.shape[1]/2)
    for i in range(iC-radius, iC+radius):
        for j in range(jC-radius, jC+radius):
            if (np.sqrt((i-iC)**2+(j-jC)**2)) <= radius:
                x = random.choice([-1, 1])
                grid_0[i,j] = x
    return grid_0
        
#bounds_set = set()
# Initial squre population
def fill(grid, a):
    grid_0 = np.copy(grid)
    xlen = grid.shape[0]#need -1?
    ylen = grid.shape[1]
    for x in range(int(a*xlen), int(xlen-a*xlen)):
        for y in range(int(a*ylen), int(ylen-a*ylen)):
            i = random.choice([-1, 1])
            grid_0[x,y] = i
    return grid_0

# Initialise boundary array
def init_boundary_arr(grid):
    xlen = grid.shape[0]-1
    ylen = grid.shape[1]-1
    b_cell = []
    for i in range(xlen):
        for j in range(ylen):
            neigh = neighbors(i, j, grid)
            neigh_occupancy = [grid[neigh[n][0], neigh[n][1]] for n in range(len(neigh))]
            #np.any --> if you have one false inside, and if you have something true 
            if np.any((np.asarray(neigh_occupancy)!=0)) and np.any((np.asarray(neigh_occupancy)==0)) and (grid[i,j]!=0):
                b_cell.append([i,j]) 
                #bounds_set.add((i, j))
    return b_cell

# Choose a random cell
def randomcell(grid):
    x = random.randint(0,grid.shape[0]-1)
    y = random.randint(0,grid.shape[1]-1)
    
    return x,y

# Choose a random cell at the boundary
def random_b_cell(bounds):
    x = random.randint(0,len(bounds)-1)
    
    return bounds[x]

# Probability of growth
# returns 1 if it divided (growth), 0 if not (death)
def flipg(x, y, grid, growth_rate_vec):
    cell = grid[x,y]
    p = growth_rate_vec[int(cell+2)]
    
    return 1 if random.random() < p else 0

# Define how to pick up mutations
def flipm(x, y, grid, p_mu):
    cell = grid[x,y]
    if cell == 1:
        return 1 if random.random() < float(p_mu[0]) else 0
    elif cell == -1:
        return 1 if random.random() < float(p_mu[1]) else 0

# Mutation pathway
def mutate(x, y, grid):
    cell = grid[x,y]
    if cell == 1:
        grid[x,y] = 2
    elif cell == -1:
        grid[x,y] = -2
        
    return grid
    
# Define neighbors (diagonals are second neighbors) 
def neighbors(x, y, grid):
    xlen = grid.shape[0] - 1
    ylen = grid.shape[1] - 1
        
    # Two neighbors    
    if x == 0 and y == 0:
        neighbors = [[0,1],[1,0]]
    elif x == xlen and y == ylen:
        neighbors = [[xlen-1,ylen],[xlen,ylen-1]]
        
    # Three neighbors    
    elif x == xlen:
        neighbors = [[xlen, y-1],[xlen,y+1],[xlen-1,y]]
    elif y == ylen: 
        neighbors = [[x-1,ylen],[x+1,ylen],[x,ylen-1]]
    elif x == 0:
        neighbors = [[0,y+1],[0,y-1],[1,y]]
    elif y == 0:
        neighbors = [[x+1,0],[x-1,0],[x,1]]
        
    # Four neighbors    
    else:
        neighbors = [[x,y+1],[x,y-1],[x+1,y],[x-1,y]]
        
    return neighbors


# Divides cell, adds new edge cells to boundary arr, 
# deletes cells that are no longer on boundary
def divide(x, y, grid, bounds):
    neigh = neighbors(x,y,grid)
    children = []
    for i in range(len(neigh)):
        curr_x = neigh[i][0]
        curr_y = neigh[i][1]
        if grid[curr_x, curr_y] == 0:
            grid[curr_x, curr_y] = grid[x, y]
            children.append([curr_x, curr_y])
            bounds.append([curr_x, curr_y])
            curr_neigh = neighbors(curr_x, curr_y, grid)
            curr_neigh_occ = [grid[curr_neigh[n][0], curr_neigh[n][1]] for n in range(len(curr_neigh))]
            #if a newly filled cell has no empty neighbors, then delete from the bounds array
            #if all filled, np.all returns true
            if np.all((np.asarray(curr_neigh_occ)!=0)):
                bounds = remove(curr_x, curr_y, bounds)
                
    return(children)

def remove(x, y, bounds):
    bounds.pop(bounds.index([x, y]))
    return bounds 

# Use boundary as outcome --> compute the ratio of cell phenotypes 
# on outer edge after loop finishes
def edge_ratio(bounds, grid):
    cell_count_vec = [0, 0, 0, 0, 0]
    final_bounds = []
    for i in range(len(bounds)):
        x = bounds[i][0]
        y = bounds[i][1]
        neigh = neighbors(x, y, grid)
        neigh_occupancy = [grid[neigh[n][0], neigh[n][1]] for n in range(len(neigh))]
        if np.any((np.asarray(neigh_occupancy)!=0)) and np.any((np.asarray(neigh_occupancy)==0)) and (grid[x,y]!=0):
            final_bounds.append([x, y])
            cell = grid[x,y]
            cell_count_vec[int(cell+2)] = cell_count_vec[int(cell+2)] + 1
    
    total_n = cell_count_vec[1] + cell_count_vec[3] + cell_count_vec[2] + cell_count_vec[0] + cell_count_vec[4]
    if cell_count_vec[1] != 0:
        #ratio = cell_count_vec[4]/cell_count_vec[1]/total_n # mut/non mut (2/-1)
        ratio = cell_count_vec[4]/cell_count_vec[1] # mut/non mut (2/-1)
    if cell_count_vec[1] == 0:
        #ratio = cell_count_vec[4]/total_n
        ratio = cell_count_vec[4]
    
    return cell_count_vec, final_bounds, ratio

def fitness_diff(df1,df2):
    NM = 1-df2
    M = NM+df1   
    return NM,M


def load_growth_rates(fne, pyr_level_idx):
	data_fitted = np.loadtxt(fne)

	pyr = np.array([0, 150, 300, 500, 800, 1000, 1500], dtype=float)
	pyr_lin = np.linspace(min(pyr), max(pyr), num=10000)
	for i in range(6):
		if i == 0:
			f_318_WT = data_fitted[i*10000: (i+1)*10000]
		if i == 4:
			f_611_WT = data_fitted[i*10000: (i+1)*10000]
		if i == 5:
			f_611_mut = data_fitted[i*10000: (i+1)*10000]

	# Vector setting difference in growth rates 
	# growth_rate_vec = [_, _, _, _, _] --> corresponds to cell -2,-1,0,1,2
	# --> use to set highest growth rate to 1, change all others proportionally
	x = 1/max(f_318_WT[pyr_level_idx],f_611_WT[pyr_level_idx],f_611_mut[pyr_level_idx])
	print('318 WT: ', pyr_lin[pyr_level_idx], f_318_WT[pyr_level_idx]*x) 
	print('611 WT: ',pyr_lin[pyr_level_idx], f_611_WT[pyr_level_idx]*x) 
	print('611 mut: ',pyr_lin[pyr_level_idx], f_611_mut[pyr_level_idx]*x) 
	growth_rate_vec = [f_611_mut[pyr_level_idx], f_318_WT[pyr_level_idx], 0, f_611_WT[pyr_level_idx], f_611_mut[pyr_level_idx]]

	return growth_rate_vec, f_318_WT, f_611_mut, f_611_WT
