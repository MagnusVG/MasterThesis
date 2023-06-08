#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import time
import argparse

class MyDataset(Dataset):
    def __init__(self, array, target, balance_set=False):         
        self.array = array
        self.target = target # np array (1.)        

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
        
        X = self.array[index]        

        # Cast to 32 bit for GPU training
        X = np.float32(np.array(X))        
        target = np.expand_dims(np.float32(self.target[index]),-1)

        return {"point": X, "target": target}

    def __len__(self):
        l = int(len(self.target))
        return int(l)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, input_dim)
        self.layer_2 = nn.Linear(input_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.sigmoid(self.layer_2(x))
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
        elif validation_loss > (self.min_validation_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

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
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save', type=str)
    args = parser.parse_args()

    # Read data and filter edges
    read_train1 = pd.read_csv("../project_data/ac_felt3_clean.csv")
    read_train2 = pd.read_csv("../project_data/ac_felt4.csv")
    read_validation = pd.read_csv("../project_data/ac_felt1.csv")

    # Normalize
    normalized_train1 = (read_train1-read_train1.min())/(read_train1.max()-read_train1.min())
    normalized_train2 = (read_train2-read_train2.min())/(read_train2.max()-read_train2.min())
    normalized_train = pd.concat([normalized_train1, normalized_train2])
    normalized_validation = (read_validation-read_validation.min())/(read_validation.max()-read_validation.min())

    train_data = np.asarray(normalized_train[normalized_train.columns[0:-1]])
    train_labels = np.asarray(normalized_train["accepted"])
    validation_data = np.asarray(normalized_validation[normalized_validation.columns[0:-1]])
    validation_labels = np.asarray(normalized_validation["accepted"])

    device = torch.device("cuda")
    kwargs = {'num_workers': 3, 'pin_memory': True}

    # Tensor data
    train_dataset = MyDataset(train_data, train_labels, balance_set=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    validation_dataset = MyDataset(validation_data, validation_labels)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # Create model
    input_shape = len(normalized_train.columns[0:-1])
    model = NeuralNetwork(input_shape, 1).to(device)

    learning_rate = 0.000008
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # SGD

    # Early stopper
    early_stopping = True
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)

    # Dynamic lr
    #scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=20)
    dynamic_scheduler = False
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    training_started = time.time()

    train_loss = []
    validation_loss = []
    for epoch in range(args.epochs):
        epoch_started = time.time()
        print('Epoch ' + str(epoch) + ' started...')

        if dynamic_scheduler:
            before_lr = optimizer.param_groups[0]["lr"]
            print(f"Current learning rate: {before_lr} \n")

        # Training
        epoch_train_loss = 0
        for train_data in train_dataloader:
            point, target = train_data["point"].to(device), train_data["target"].to(device)

            # Predict
            pred = model(point)
            loss = loss_fn(pred, target)
            loss_item = loss.item()

            # Store loss
            train_loss.append(loss_item)

            # Print per epoch
            epoch_train_loss += loss_item

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train_loss /= len(train_dataloader)
        print(f"Train Error: \n Avg loss: {epoch_train_loss:>8f} \n")

        # Dynamic learning rate 
        if dynamic_scheduler:
            scheduler.step()

        # Validation
        epoch_val_loss = 0
        with torch.no_grad():
            for validation_data in validation_dataloader:
                point, target = validation_data["point"].to(device), validation_data["target"].to(device)

                # Predict
                pred = model(point)
                loss = loss_fn(pred, target)
                loss_item = loss.item()
                
                # Store loss
                validation_loss.append(loss_item)

                # Print per epoch
                epoch_val_loss += loss_item
        epoch_val_loss /= len(validation_dataloader)
        print(f"Validation Error: \n Avg loss: {epoch_val_loss:>8f} \n")

        epoch_finished = (time.time() - epoch_started)
        print('Epoch used ' + str(epoch_finished) + ' seconds\n')
        print("-------------------------------")

        # Early stopping
        if early_stopping:
            if early_stopper.early_stop(epoch_val_loss):
                print("Stopping early..")         
                break

    training_finished = (time.time() - training_started)
    print('Execution time in seconds: ' + str(training_finished))

    np.save("models/ac/train_loss"+args.save+".npy", np.array(train_loss))
    np.save("models/ac/validation_loss"+args.save+".npy", np.array(validation_loss))

    plot_result(train_loss, args.epochs, "models/ac/train_loss_plot"+args.save+".jpg")
    plot_result(validation_loss, args.epochs, "models/ac/validation_loss_plot"+args.save+".jpg")
    
    model.save("models/ac/model"+args.save+".pt")

if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()
