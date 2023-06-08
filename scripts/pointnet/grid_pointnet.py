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

def get_single_point_clouds(dataframe, num_points):
    negatives = np.array(dataframe[dataframe["accepted"] == 0])
    negatives_index = dataframe[dataframe["accepted"] == 0].index.to_numpy()
    point_cloud_data = []
    point_cloud_labels = []
    point_cloud_indexes = []
    for idx, negative in enumerate(negatives):
        x, y, z, accepted = negative

        # Find area around
        cs = 1
        negative_box = area_of_dataframe(dataframe, x-cs, x+cs, y-cs, y+cs)
        while len(negative_box[negative_box["accepted"] == 1]) < num_points:
            cs += 0.5
            negative_box = area_of_dataframe(dataframe, x-cs, x+cs, y-cs, y+cs)

        # Select negatives
        point = negative_box[negative_box.index == negatives_index[idx]]

        # Sample remaining positives
        positives = negative_box[negative_box["accepted"] == 1]
        sample_positives = positives.sample(n=num_points-1)

        # Join together
        combine = pd.concat((point, sample_positives))

        point_cloud_data.append(combine[["x", "y", "z"]])
        labels = combine[["accepted"]]
        point_cloud_labels.append(labels)
        point_cloud_indexes.append(combine.index.to_numpy())

    return point_cloud_data, point_cloud_labels, point_cloud_indexes

def get_random_point_clouds(dataframe, n_points, cell_size, part):
    lower = ((part-1)*0.1)*dataframe.x.max()
    upper = ((part)*0.1)*dataframe.x.max()
    
    cells_in_x = np.arange(lower, upper+cell_size, cell_size)
    cells_in_y = np.arange(dataframe.y.min(), dataframe.y.max()+cell_size, cell_size)
    
    point_cloud_data = []
    point_cloud_labels = []
    point_cloud_indexes = []
    for ix, x in enumerate(cells_in_x):
        for iy, y in enumerate(cells_in_y):
            if ix != 0 and iy != 0:
                lowX, highX = cells_in_x[ix-1], x
                lowY, highY = cells_in_y[iy-1], y
                cell = area_of_dataframe(dataframe, lowX, highX, lowY, highY)

                # This is a cell that is too small
                while len(cell) < n_points:
                    if lowX > 0.5: lowX -= 0.5
                    if lowY > 0.5: lowY -= 0.5
                    if highX < dataframe.x.max()-0.5: highX += 0.5
                    if highY < dataframe.y.max()-0.5: highY += 0.5
                    cell = area_of_dataframe(dataframe, lowX, highX, lowY, highY)

                # Handles several samplings if necessary
                original_cell = cell.copy(deep=True)
                while len(cell) >= n_points:
                    # Take a sample from the cell
                    point_cloud = cell.sample(n=n_points)
                    # Get indexes for the remaining points in the cell
                    pc_indexes = point_cloud.index.to_numpy()
                    all_indexes = cell.index.to_numpy()
                    remaining_indexes = [index for index in all_indexes if index not in pc_indexes]
                    # Add data to return lists
                    point_cloud_data.append(point_cloud[["x", "y", "z"]])
                    point_cloud_labels.append(point_cloud[["accepted"]])
                    point_cloud_indexes.append(pc_indexes)
                    # Get the point cloud and the remaining point cloud
                    cell = cell.filter(items=remaining_indexes, axis=0)

                # Now there are less than 1024, take the remaining + something from the others
                pc_indexes = cell.index.to_numpy()
                all_indexes = original_cell.index.to_numpy()
                remaining_indexes = [index for index in all_indexes if index not in pc_indexes]
                remaining_cell = original_cell.filter(items=remaining_indexes, axis=0)
                remaining_sample = remaining_cell.sample(n=n_points-len(cell))
                combine = pd.concat((cell, remaining_sample))
                point_cloud_data.append(combine[["x", "y", "z"]])
                point_cloud_labels.append(combine[["accepted"]])
                point_cloud_indexes.append(combine.index.to_numpy())

    return point_cloud_data, point_cloud_labels, point_cloud_indexes

def main():
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--num_points', type=int)
    parser.add_argument('--random_sampling', type=int)
    parser.add_argument('--part', type=int)
    args = parser.parse_args()
    
    # Start timer
    startTime = time.time()

    TRAIN_CELL_SIZE = 1
    VAL_CELL_SIZE = 2
    if args.num_points == 256:
        TRAIN_CELL_SIZE = 1
        VAL_CELL_SIZE = 2
    if args.num_points == 512:
        TRAIN_CELL_SIZE = 1.5
        VAL_CELL_SIZE = 3
    if args.num_points == 1024:
        TRAIN_CELL_SIZE = 2
        VAL_CELL_SIZE = 4
    if args.num_points == 2048:
        TRAIN_CELL_SIZE = 2.5
        VAL_CELL_SIZE = 6

    # Read training data
    read_train1 = pd.read_csv("../project_data/ac_felt3_clean.csv")
    #read_train1 = pd.read_csv("../project_data/pn_felt1_EW.csv")
    #read_train2 = pd.read_csv("../project_data/pn_felt1_NS.csv")
    read_train1 = read_train1[["x", "y", "z", "accepted"]]
    if args.random_sampling == 0:
        train_data1, train_labels1, train_indexes1 = get_single_point_clouds(read_train1, args.num_points)
    #    train_data2, train_labels2, train_indexes2 = get_single_point_clouds(read_train2, args.num_points)
    else:
        train_data1, train_labels1, train_indexes1 = get_random_point_clouds(read_train1, args.num_points, TRAIN_CELL_SIZE, args.part)
    #    train_data2, train_labels2, train_indexes2 = get_random_point_clouds(read_train2, args.num_points, TRAIN_CELL_SIZE)
    #train_data = np.concatenate((train_data1, train_data2))
    #train_labels = np.concatenate((train_labels1, train_labels2))
    #train_indexes = np.concatenate((train_indexes1, train_indexes2))

        # Read validation data
    #read_validation = pd.read_csv("../project_data/pn_felt2_NS.csv")
    #val_data, val_labels, val_indexes = get_random_point_clouds(read_validation, args.num_points, TRAIN_CELL_SIZE)

    # Save it
    folder_str = "random/parts"
    #if args.random_sampling == 0:
    #    folder_str = "single"
    #else:
    #    folder_str = "random"
    np.save("grids/points_"+str(args.num_points)+"/"+folder_str+"/train_data_part"+str(args.part)+".npy", train_data)
    np.save("grids/points_"+str(args.num_points)+"/"+folder_str+"/train_labels_part"+str(args.part)+".npy", train_labels)
    np.save("grids/points_"+str(args.num_points)+"/"+folder_str+"/train_indexes_part"+str(args.part)+".npy", train_indexes)
    #np.save("grids/points_"+str(args.num_points)+"/"+folder_str+"/val_data.npy", val_data)
    #np.save("grids/points_"+str(args.num_points)+"/"+folder_str+"/val_labels.npy", val_labels)
    #np.save("grids/points_"+str(args.num_points)+"/"+folder_str+"/val_indexes.npy", val_indexes)

    # Get executiontime
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

if __name__ == '__main__':
    main()
