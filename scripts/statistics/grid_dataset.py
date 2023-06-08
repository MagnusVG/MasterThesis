#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import time
import argparse

def area_of_dataframe(dataframe, x_start, x_last, y_start, y_last):
    area_dataframe = dataframe[
        (dataframe["x"] >= x_start) &
        (dataframe["x"] < x_last) & 
        (dataframe["y"] >= y_start) & 
        (dataframe["y"] < y_last)
    ]
    return area_dataframe

def cell_statistics(cell, z_index):
    cell_z = cell[:, z_index]
    median = np.median(cell_z)
    std = np.std(cell_z)
    mx = np.max(cell_z)
    mn = np.min(cell_z)

    return median, std, mx, mn

def dataframe_to_normalized_stats_grid(dataframe, cell_size, z_index, part):
    # Out area always starts at 0 from XYZ preprocessing
    lower = ((part-1)*0.1)*dataframe.x.max()
    upper = ((part)*0.1)*dataframe.x.max()
    cells_in_x = np.arange(lower, upper+cell_size, cell_size)
    cells_in_y = np.arange(dataframe.y.min(), dataframe.y.max()+cell_size, cell_size)

    # Min-max for normalizing
    z_min = dataframe.z.min()
    z_max = dataframe.z.max()

    x_grid = []
    # Iterate over cells
    for ix, x in enumerate(cells_in_x):
        y_grid = []
        for iy, y in enumerate(cells_in_y):
            if ix != 0 and iy != 0:
                cell = np.array(area_of_dataframe(dataframe, cells_in_x[ix-1], x, cells_in_y[iy-1], y))

                if len(cell) <= 0:
                    # If no points in cell, add -1 stats
                    stats = [-1, -1, -1, -1]
                    y_grid.append(stats)
                else:
                    # Normalize
                    #norm_cell = (cell - sample_min) / (sample_max - sample_min) 
                    norm_cell = np.copy(cell)
                    norm_cell[:, z_index] = (norm_cell[:, z_index] - z_min) /(z_max - z_min) 

                    # Get cell statistics
                    median, std, mx, mn = cell_statistics(norm_cell, z_index)
                    stats = [mn, median, mx, std]

                    # Add to grid
                    y_grid.append(stats)
        if len(y_grid) != 0:
            x_grid.append(y_grid)

    return np.asarray(x_grid)

def main():
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--cell_size', type=float)
    parser.add_argument('--folder', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--part', type=int)
    args = parser.parse_args()
    
    # Start timer
    startTime = time.time()

    # Define statistics and size
    CELL_SIZE = args.cell_size
    Z_INDEX = 2 # Where to find the z value

    # Read data
    data = pd.read_csv("../project_data/"+args.dataset)

    #data = data.rename(columns={"X":"x", "Y":"y", "Z":"z", "Accepted":"accepted"})

    # Turn area into cells
    stat_grid = dataframe_to_normalized_stats_grid(data, CELL_SIZE, Z_INDEX, args.part)

    # Save it
    np.save("../project_data/"+args.folder+"_part"+str(args.part)+".npy", stat_grid)

    # Get executiontime
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

if __name__ == '__main__':
    main()
