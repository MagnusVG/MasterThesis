#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import time
import argparse

def xyz_to_dataframe(accepted_xyz, rejected_xyz):
    COLUMNS = [
        "x", "y", "z", "longitude", "latitude", "date", "time",
        "quality", "travelTime", "svl", "bearingAngle", "tiltAngle",
        "filename", "beamNumber", "scanNumber", "thu", "tvu", "accepted"
    ]
    #COLUMNS = ["x", "y", "z", "accepted"]
    read_data = []
    dataframes = []

    # Rejected data
    with open(rejected_xyz, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            columns = line.strip().split(" ")
            columns.append(0)
            read_data.append(np.array(columns))

            if (idx != 0) and ((idx % 100000) == 0):
                dataframe = pd.DataFrame(read_data, columns=COLUMNS)
                dataframes.append(dataframe)
                read_data = []
        dataframe = pd.DataFrame(read_data, columns=COLUMNS)
        dataframes.append(dataframe)


    # Accepted data
    with open(accepted_xyz, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            columns = line.strip().split(" ")
            columns.append(1)
            read_data.append(np.array(columns))

            if (idx != 0) and ((idx % 500000) == 0):
                dataframe = pd.DataFrame(read_data, columns=COLUMNS)
                dataframes.append(dataframe)
                read_data = []
        dataframe = pd.DataFrame(read_data, columns=COLUMNS)
        dataframes.append(dataframe)

    # Combine all dataframes
    data = pd.concat(dataframes, ignore_index=True)

    # Convert to numeric
    data["x"] = pd.to_numeric(data["x"])
    data["y"] = pd.to_numeric(data["y"])
    data["z"] = pd.to_numeric(data["z"])

    data["x"] -= np.min(data["x"])
    data["y"] -= np.min(data["y"])
    data["z"] -= np.min(data["z"])

    data = data.sort_values(by=["x", "y", "z"], ascending=True)

    return data

def main():
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--felt_akseptert', type=str)
    parser.add_argument('--felt_forkastet', type=str)
    parser.add_argument('--felt_output', type=str)
    args = parser.parse_args()
    
    data_folder = "../project_data/"
    felt_accepted = data_folder + args.felt_akseptert
    felt_rejected = data_folder + args.felt_forkastet
    felt_output = data_folder + args.felt_output
    
    # Start timer
    startTime = time.time()

    # Extract data from xyz
    felt_data = xyz_to_dataframe(felt_accepted, felt_rejected)

    # Convert to csv
    felt_data.to_csv(felt_output, index=False)

    # Get executiontime
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))

if __name__ == '__main__':
    main()
