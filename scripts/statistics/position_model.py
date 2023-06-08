#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import time
import argparse

class MyDataset(Dataset):
    def __init__(self, image, array, target, cell_size, grid_size=1, balance_set=False):
        self.image = image            
        self.array = array
        self.target = target    
        self.cell_size = cell_size
        self.grid_size = grid_size 

        self.maxx, self.maxy, self.numz = self.image.shape # size of data-block

        self.balance_set = balance_set

        if balance_set:
            self.positive = np.argwhere(self.target > 0.5)
            self.negative = np.argwhere(self.target <= 0.5)

    def __getitem__(self, idx):

        index = idx

        # Balance by alternating between random positive and negative sample
        if self.balance_set:
            if index % 2 == 0:
                index = self.positive[np.random.randint(len(self.positive))][0]
            else:
                index = self.negative[np.random.randint(len(self.negative))][0]
        
        vals = self.array[index]        
        x, y, z = vals

        # Create X vector
        X = [z] # Removed position from array, are they needed here?

        # - cw, cy : find cell position based on x and y
        cw = int(x / self.cell_size)
        ch = int(y / self.cell_size)

        # Create Y vector of desired size with default -1 values
        Y = np.full((2*self.grid_size + 1, 2*self.grid_size + 1, 4), -1.)

        # Prepare slicing parameters
        fx0, fx1 = cw - self.grid_size, cw + self.grid_size + 1
        fy0, fy1 = ch - self.grid_size, ch + self.grid_size + 1
        tx0 = -fx0 if fx0 < 0 else 0
        ty0 = -fy0 if fy0 < 0 else 0
        tx1 = 2*self.grid_size + 1
        ty1 = 2*self.grid_size + 1
        fx0 = max(0, fx0)
        fy0 = max(0, fy0)
        if fx1 > self.maxx:
            tx1 = (2*self.grid_size + 1) - (fx1 - self.maxx)
        if fy1 > self.maxy:
            ty1 = (2*self.grid_size + 1) - (fy1 - self.maxy)
        fx1 = min(self.maxx, fx1)
        fy1 = min(self.maxy, fy1)

        # Slice Y vector
        Y[tx0:tx1,ty0:ty1] = self.image[fx0:fx1, fy0:fy1]

        # Reshape
        Y = Y.reshape([4, self.grid_size*2+1, self.grid_size*2+1])

        # Cast to 32 bit for GPU training
        Y = np.float32(Y)
        X = np.float32(np.array(X))
        target = np.expand_dims(np.float32(self.target[index]),-1)

        return {"point": X, "grid" : Y, "target" : target}

    def __len__(self):
        l = int(len(self.target))
        return int(l)
    
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, int(input_dim*0.5))
        self.layer_2 = nn.Linear(int(input_dim*0.5), int(input_dim*0.25))
        self.layer_3 = nn.Linear(int(input_dim*0.25), 1)
       
    def forward(self, point, grid):
        grid = torch.flatten(grid, 1) # Flatten the grid
        x = torch.cat((point, grid), 1) # Combine with point

        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = torch.sigmoid(self.layer_3(x))

        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

class ConvNeuralNetwork(nn.Module):
    def __init__(self, grid_size):
        super(ConvNeuralNetwork, self).__init__()

        kernel1, stride1, pool1 = 3, 1, 2
        kernel2, stride2, pool2 = 5, 1, 2
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=kernel1, stride=stride1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=pool1)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=kernel2, stride=stride2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=pool2)
        )

        gs = calculate_conv((grid_size*2)+1, kernel1, stride1)
        gs = calculate_pool(gs, pool1)
        gs = calculate_conv(gs, kernel2, stride2)
        gs = calculate_pool(gs, pool2)
        input_size = gs*gs*16
        input_size = input_size.astype(np.int64)

        self.fc_layer1 = nn.Sequential(
            nn.Linear(input_size+1, int(input_size*(50/100))),
            nn.ReLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(int(input_size*(50/100)), int(input_size*(25/100))),
            nn.ReLU()
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(int(input_size*(25/100)), 1),
            nn.Sigmoid()
        )
       
    def forward(self, point, grid):
        grid = self.conv_layer1(grid)
        grid = self.conv_layer2(grid)

        grid = torch.flatten(grid, 1) # flatten all dimensions except batch
        x = torch.cat((point, grid), 1)

        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)

        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def calculate_conv(grid_size, kernel, stride, padding=0):
    return np.floor(((grid_size + 2*padding - kernel) / stride) + 1)

def calculate_pool(grid_size, kernel):
    return np.floor(grid_size/kernel)

def plot_result(loss, epochs, save_location):
    plt.clf()

    step = np.linspace(0, epochs, len(loss))
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(step, np.array(loss))
    plt.title("Step-wise Loss", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.savefig(save_location)

def main():
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--grid_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--cell_size', type=float)
    parser.add_argument('--grid_folder', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--convolution', type=int)
    args = parser.parse_args()

    # Read data and filter edges
    read_train = pd.read_csv("../project_data/ac_felt1.csv")
    read_validation = pd.read_csv("../project_data/ac_felt2_validation.csv")
    train_grid = np.load("../project_data/"+args.grid_folder+"/ac_felt1.npy")
    validation_grid = np.load("../project_data/"+args.grid_folder+"/ac_felt2_validation.npy")
    
    # Normalize train and validation
    data = read_train.copy()
    data["z"] = (data.z - np.min(data.z)) / (np.max(data.z) - np.min(data.z))

    validation = read_validation.copy()
    validation["z"] = (validation.z - np.min(validation.z)) / (np.max(validation.z) - np.min(validation.z))

    train_data = np.asarray(data[["x", "y", "z"]])
    train_labels = np.asarray(data["accepted"])
    validation_data = np.asarray(validation[["x", "y", "z"]])
    validation_labels = np.asarray(validation["accepted"])

    device = torch.device("cuda")
    kwargs = {'num_workers': 3, 'pin_memory': True}

    # Tensor data
    train_dataset = MyDataset(train_grid, train_data, train_labels, cell_size=args.cell_size, grid_size=args.grid_size, balance_set=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    validation_dataset = MyDataset(validation_grid, validation_data, validation_labels, cell_size=args.cell_size,  grid_size=args.grid_size)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Create model
    conv_str = ""
    if args.convolution == 1: # With grid size=5, use lr=?
        print("Convolution model")
        conv_str = "conv_"
        model = ConvNeuralNetwork(args.grid_size).to(device)
    else: # With grid size=1, use lr=0.001
        print("Normal model")
        input_shape = (np.sum([8*i for i in range(args.grid_size+1)])*4) + 4 + 1
        model = NeuralNetwork(input_shape).to(device)

    learning_rate = args.lr
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Early stopper
    use_earlyStop = True
    early_stopper = EarlyStopper(patience=10, min_delta=0.01)

    # Learning rate
    use_scheduler = False
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    training_started = time.time()

    train_loss = []
    validation_loss = []
    for epoch in range(args.epochs):
        epoch_started = time.time()
        print('Epoch ' + str(epoch) + ' started...')

        if use_scheduler:
            before_lr = optimizer.param_groups[0]["lr"]
            print(f"Current learning rate: {before_lr} \n")

        # Training
        epoch_train_loss = 0
        for train_data in train_dataloader:
            point, grid, target = train_data["point"].to(device), train_data["grid"].to(device), train_data["target"].to(device)

            # Predict
            pred = model(point, grid)
            loss = loss_fn(pred, target)

            # Store loss
            train_loss.append(loss.item())

            # Print per epoch
            epoch_train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train_loss /= len(train_dataloader)
        print(f"Train Error: \n Avg loss: {epoch_train_loss:>8f} \n")

        # Dynamic learning rate 
        if use_scheduler:
            scheduler.step()

        # Validation
        epoch_val_loss = 0
        with torch.no_grad():
            for validation_data in validation_dataloader:
                point, grid, target = validation_data["point"].to(device), validation_data["grid"].to(device), validation_data["target"].to(device)

                # Predict
                pred = model(point, grid)
                loss = loss_fn(pred, target)
                
                # Store loss
                validation_loss.append(loss.item())

                # Print per epoch
                epoch_val_loss += loss.item()
        epoch_val_loss /= len(validation_dataloader)
        print(f"Validation Error: \n Avg loss: {epoch_val_loss:>8f} \n")

        epoch_finished = (time.time() - epoch_started)
        print('Epoch used ' + str(epoch_finished) + ' seconds\n')
        print("-------------------------------")

        # Early stopping
        if use_earlyStop and epoch > 10:
            if early_stopper.early_stop(epoch_val_loss):
                print("Stopping early..")  
                break

    training_finished = (time.time() - training_started)
    print('Execution time in seconds: ' + str(training_finished))

    np.save("models/"+args.grid_folder+"/grid_size_"+str(args.grid_size)+"/"+conv_str+"train_loss"+args.save+".npy", np.array(train_loss))
    np.save("models/"+args.grid_folder+"/grid_size_"+str(args.grid_size)+"/"+conv_str+"validation_loss"+args.save+".npy", np.array(validation_loss))

    plot_result(train_loss, args.epochs, "models/"+args.grid_folder+"/grid_size_"+str(args.grid_size)+"/"+conv_str+"train_loss_plot"+args.save+".jpg")
    plot_result(validation_loss, args.epochs, "models/"+args.grid_folder+"/grid_size_"+str(args.grid_size)+"/"+conv_str+"validation_loss_plot"+args.save+".jpg")

    model.save("models/"+args.grid_folder+"/grid_size_"+str(args.grid_size)+"/"+conv_str+"model_dict"+args.save+".pt")
    torch.save(model, "models/"+args.grid_folder+"/grid_size_"+str(args.grid_size)+"/"+conv_str+"model"+args.save+".pt")
    
if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()
