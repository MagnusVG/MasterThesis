#!/usr/bin/env python
# coding=utf-8

import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
from sklearn.utils import class_weight


def area_of_dataframe(dataframe, x_start, x_last, y_start, y_last):
    area_dataframe = dataframe[
        (dataframe["X"] >= x_start) &
        (dataframe["X"] < x_last) & 
        (dataframe["Y"] >= y_start) & 
        (dataframe["Y"] < y_last)
    ]
    return area_dataframe

def binary_labels(x):
    if x == 0:
        return [1, 0]
    return [0, 1]

def get_point_clouds(dataframe, num_points):
    negatives = np.array(dataframe[dataframe["Accepted"] == 0])
    negatives_index = dataframe[dataframe["Accepted"] == 0].index.to_numpy()
    point_cloud_data = []
    point_cloud_labels = []
    point_cloud_indexes = []
    for idx, negative in enumerate(negatives):
        x, y, z, accepted = negative

        # Find area around
        cs = 1
        negative_box = area_of_dataframe(dataframe, x-cs, x+cs, y-cs, y+cs)
        while len(negative_box[negative_box["Accepted"] == 1]) < num_points:
            cs += 1
            negative_box = area_of_dataframe(dataframe, x-cs, x+cs, y-cs, y+cs)

        # Select negatives
        point = negative_box[negative_box.index == negatives_index[idx]]

        # Sample remaining positives
        positives = negative_box[negative_box["Accepted"] == 1]
        sample_positives = positives.sample(n=num_points-1)

        # Join together
        combine = pd.concat((point, sample_positives))

        point_cloud_data.append(combine[["X", "Y", "Z"]])
        labels = combine[["Accepted"]]
        point_cloud_labels.append(labels)
        point_cloud_indexes.append(combine.index.to_numpy())

    return point_cloud_data, point_cloud_labels, point_cloud_indexes

def get_test_point_clouds(dataframe, cell_size, n_points):
    cells_in_x = np.arange(dataframe.X.min(), dataframe.X.max()+cell_size, cell_size)
    cells_in_y = np.arange(dataframe.Y.min(), dataframe.Y.max()+cell_size, cell_size)
    
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
                    if highX < dataframe.X.max()-0.5: highX += 0.5
                    if highY < dataframe.Y.max()-0.5: highY += 0.5
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
                    point_cloud_data.append(point_cloud[["X", "Y", "Z"]])
                    point_cloud_labels.append(point_cloud[["Accepted"]])
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
                point_cloud_data.append(combine[["X", "Y", "Z"]])
                point_cloud_labels.append(combine[["Accepted"]])
                point_cloud_indexes.append(combine.index.to_numpy())

    return point_cloud_data, point_cloud_labels, point_cloud_indexes


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PointNetData(Dataset):
    def __init__(self, point_clouds, labels, indexes):
        self.point_clouds = point_clouds # (X, 1024, 3)
        self.labels = labels # (X, 1024, 2)
        self.indexes = indexes

    def __getitem__(self, idx):
        point_set = np.array(self.point_clouds[idx]).astype(np.float32)
        labels = np.array(self.labels[idx]).astype(np.float32) # float32 
        indexes = np.array(self.indexes[idx]).astype(np.int32)

        norm_point_set = pc_normalize(point_set)

        return norm_point_set, labels, indexes, point_set

    def __len__(self):
        length = int(len(self.labels))
        return length
    
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
    
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    S = npoint
    centroids = torch.zeros(B, S, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(S):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    K = nsample
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:,:,:K]
    group_first = group_idx[:,:,0].view(B, S, 1).repeat([1, 1, K])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx
    
def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint

    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz -= new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
    
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
    
    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
    
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:,:,:3], idx[:,:,:3] #[B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists #[B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1) #[B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim = 2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
    
class PointNet2PartSeg(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2PartSeg, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 64, 3, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 256, 1024], True)
        self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

class CustomBCELoss():
    def __init__(self, weights):
        self.weights = weights

    def train_loss(self, data, target):
        data = torch.clamp(data, min=1e-7, max=1-1e-7)
        bce = - self.weights[1] * target * torch.log(data) - (1 - target) * self.weights[0] * torch.log(1 - data)
        return torch.mean(bce)

    def validation_loss(self, data, target):
        data = torch.clamp(data, min=1e-7, max=1-1e-7)
        bce = - target * torch.log(data) - (1 - target) * torch.log(1 - data)
        return torch.mean(bce)

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

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
    
def main():
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_points', type=int)
    parser.add_argument('--random_sampling', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save', type=str)
    parser.add_argument('--normalize', type=int)
    args = parser.parse_args()


    BATCH_SIZE = args.batch_size
    N_POINTS = args.num_points
    EPOCHS = args.epochs
    NUM_CLASSES = 1

    LR = 0.00001

    # Read training data
    read_folder = ""
    if args.random_sampling == 0:
        read_folder = "grids/points_"+str(args.num_points)+"/single/"
    else:
        read_folder = "grids/points_"+str(args.num_points)+"/random/"

    train_data = np.load(read_folder+"train_data.npy")
    if args.normalize == 1:
        norm_data = []
        for cloud in train_data:
            norm_cloud = cloud / np.linalg.norm(cloud)
            norm_data.append(norm_cloud)
        train_data = np.array(norm_data)
    train_labels = np.load(read_folder+"train_labels.npy")
    train_indexes = np.load(read_folder+"train_indexes.npy")

    # Get class weights
    #class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels.flatten()), y=train_labels.flatten())
    class_weights = [3.33, 0.59] # 15% 1000/(2*150)
    class_weights_dict = dict(enumerate(class_weights))

    # Read validation data
    val_data = np.load(read_folder+"val_data.npy")
    if args.normalize == 1:
        norm_data = []
        for cloud in val_data:
            norm_cloud = cloud / np.linalg.norm(cloud)
            norm_data.append(norm_cloud)
        val_data = np.array(norm_data)
    val_labels = np.load(read_folder+"val_labels.npy")
    val_indexes = np.load(read_folder+"val_indexes.npy")

    # Split to train/validate from both areas
    tidx = int(len(train_data)*0.9)
    vidx = int(len(val_data)*0.8)
    train_data = np.concatenate((train_data[0:tidx,:,:], val_data[0:vidx,:,:]))
    train_labels = np.concatenate((train_labels[0:tidx,:,:], val_labels[0:vidx,:,:]))
    train_indexes = np.concatenate((train_indexes[0:tidx,:], val_indexes[0:vidx,:]))
    val_data = np.concatenate((train_data[tidx:,:,:], val_data[vidx:,:,:]))
    val_labels = np.concatenate((train_labels[tidx:,:,:], val_labels[vidx:,:,:]))
    val_indexes = np.concatenate((train_indexes[tidx:,:], val_indexes[vidx:,:]))

    # Generator and Dataloader
    train_dataset = PointNetData(train_data, train_labels, train_indexes)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataset = PointNetData(val_data, val_labels, val_indexes)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = PointNet2PartSeg(NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=LR) # momentum = 0.9

    # Early stopper
    early_stopping = True
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    
    use_weights = True
    if use_weights:
        loss_fn = nn.BCELoss(reduction='none')
    else:
        loss_fn = nn.BCELoss()

    # Training
    print("Train examples: {}".format(len(train_dataset)))
    print("Evaluation examples: {}".format(len(val_dataset)))
    print("Start training...")
    cudnn.benchmark = True
    classifier.cuda()
    train_loss = []
    validation_loss = []
    for epoch in range(EPOCHS):
        print("--------Epoch {}--------".format(epoch))

        # train one epoch
        classifier.train()
        total_train_loss = 0
        for batch_idx, data in enumerate(train_dataloader, 0):
            pointcloud, label, indexes, visualize = data
            pointcloud = pointcloud.permute(0, 2, 1)
            label = label.permute(0, 2, 1)
            weights = torch.tensor(np.where(label > 0.5, class_weights_dict[1], class_weights_dict[0])).to(device)
            pointcloud, label = pointcloud.cuda(), label.cuda()

            optimizer.zero_grad()
            pred = classifier(pointcloud)
            
            loss = loss_fn(pred, label)
            if use_weights:
                loss = torch.mean(weights*loss)

            loss.backward()
            optimizer.step()

            tloss = loss.item()
            train_loss.append(tloss)
            total_train_loss += tloss
            
        print("Train loss: {:.4f}".format(total_train_loss / len(train_dataloader)))

        # eval one epoch
        classifier.eval()
        total_validation_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader, 0):
                pointcloud, label, indexes, visualize = data
                pointcloud = pointcloud.permute(0, 2, 1)
                label = label.permute(0, 2, 1)
                pointcloud, label = pointcloud.cuda(), label.cuda()

                pred = classifier(pointcloud)

                loss = loss_fn(pred, label)
                if use_weights:
                    loss = torch.mean(loss)

                vloss = loss.item()
                validation_loss.append(vloss)
                total_validation_loss += vloss
        total_validation_loss /= len(val_dataloader)
        print("Validation loss: {:.4f}".format(total_validation_loss))

        # Early stopping
        if early_stopping:
            if early_stopper.early_stop(total_validation_loss):
                print("Stopping early..")         
                break

    classifier.save("models/pn_model_dict"+args.save+".pt")
    np.save("models/train_loss"+args.save+".npy", np.array(train_loss))
    np.save("models/validation_loss"+args.save+".npy", np.array(validation_loss))
    #np.save("tuning/batch_size/"+str(BATCH_SIZE)+"/train_loss"+args.save+".npy", np.array(train_loss))
    #np.save("tuning/batch_size/"+str(BATCH_SIZE)+"/validation_loss"+args.save+".npy", np.array(validation_loss))
    
if __name__ == '__main__':
    main()
