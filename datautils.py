#!/usr/bin/env python3
"""
template.py: Describe what template does.

Created on Wed Apr  6 12:09:43 2022
@author: Sam Adams
"""

import math
import os
import json

import numpy as np
from scipy.io import loadmat
import torch
import torch.utils.data as torchd

# Dataset
class DESSImageData(torchd.Dataset):    
    def __init__(self, dessMag: torch.Tensor, dessPh: torch.Tensor,
                 t1map: torch.Tensor, t2map: torch.Tensor):
        
        if torch.is_tensor(dessMag):
            self.dessMag = dessMag
        else:
            self.dessMag = torch.tensor(dessMag)
            
        if torch.is_tensor(dessPh):
            self.dessPh = dessPh
        else:
            self.dessPh = torch.tensor(dessPh)
            
        if torch.is_tensor(t1map):
            self.t1map = t1map
        else:
            self.t1map = torch.tensor(t1map)
            
        if torch.is_tensor(t2map):
            self.t2map = t2map
        else:
            self.t2map = torch.tensor(t2map)
        
    def __len__(self):
        return self.dessMag.shape[2]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = torch.cat((self.dessMag[:, :, idx, :, :], self.dessPh[:, :, idx, :, :]), dim=2)
        y = torch.cat((self.t1map[:, :, idx:(idx+1)], self.t2map[:, :, idx:(idx+1)]),
                      dim=2)
        
        # Put channels in first dimension
        x = x.flatten(start_dim=2); x = torch.movedim(x, 2, 0)
        y = y.flatten(start_dim=2); y = torch.movedim(y, 2, 0)
        
        return x, y

# Standardization
def save_standardization_params(path: str, sourceName: str, saveName: str):
    matfile = loadmat(os.path.join(path, sourceName))
    
    mask = matfile['mask']
    dess = matfile['dess']
    t1map = matfile['t1map']
    t2map = matfile['t2map']
    
    dess_mean = np.mean(dess[np.nonzero(mask)], 0)
    dess_mean_real = np.real(dess_mean)
    dess_mean_imag = np.imag(dess_mean)
    dess_std = np.std(dess[np.nonzero(mask)], 0)
    t1_mean = np.mean(t1map[np.nonzero(mask)])
    t1_std = np.std(t1map[np.nonzero(mask)])
    t2_mean = np.mean(t2map[np.nonzero(mask)])
    t2_std = np.std(t2map[np.nonzero(mask)])
    
    standarddata = {'dess_mean_real': dess_mean_real.tolist(),
                'dess_mean_imag': dess_mean_imag.tolist(),
                'dess_std': dess_std.tolist(),
                't1_mean': t1_mean.tolist(),
                't1_std': t1_std.tolist(),
                't2_mean': t2_mean.tolist(),
                't2_std': t2_std.tolist()}
    
    with open(os.path.join(path, saveName), 'w') as fp:
        json.dump(standarddata, fp)

def split_data(allData: torchd.Dataset, batchSize: int,
               testPct: float=0.2, validPct: float=0.2)\
                -> (torchd.DataLoader, torchd.DataLoader, torchd.DataLoader):
    testCount = math.floor(len(allData) * testPct)
    validCount = math.floor(len(allData) * validPct)
    trainCount = len(allData) - (testCount + validCount)
    print(f"Number of samples: train={trainCount}, test={testCount}, validation={validCount}")

    trainDataset, testDataset, validDataset =\
        torchd.random_split(allData, (trainCount, testCount, validCount))

    trainDataLoader = torchd.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testDataLoader = torchd.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
    validDataLoader = torchd.DataLoader(validDataset, batch_size=batchSize, shuffle=False)
    
    return trainDataLoader, testDataLoader, validDataLoader

class Standardizer:
    def __init__(self, standardizationPath: str):
        with open(standardizationPath) as jsonFile:
            data = json.load(jsonFile)
        
        self.dessMean = torch.tensor(data['dess_mean_real'], dtype=torch.complex64) +\
                        1j*torch.tensor(data['dess_mean_imag'], dtype=torch.complex64)
        self.dessStd = torch.tensor(data['dess_std'], dtype=torch.float32)
        self.t1Mean = torch.tensor(data['t1_mean'], dtype=torch.float32)
        self.t1Std = torch.tensor(data['t1_std'], dtype=torch.float32)
        self.t2Mean = torch.tensor(data['t2_mean'], dtype=torch.float32)
        self.t2Std = torch.tensor(data['t2_std'], dtype=torch.float32)
    
    def standardize_features(self, x: torch.tensor) -> torch.tensor:
        out = (x - self.dessMean)/self.dessStd
        return  out
    
    def standardize_targets(self, y: torch.tensor) -> torch.tensor:
        out = torch.zeros_like(y)
        out[:, :, :, 0] = (y[:, :, :, 0] - self.t1Mean)/self.t1Std
        out[:, :, :, 1] = (y[:, :, :, 1] - self.t2Mean)/self.t2Std
        return out
    
    def destandardize_features(self, x: torch.tensor) -> torch.tensor:
        out = self.dessStd*x + self.dessMean
        return  out
    
    def destandardize_targets(self, y: torch.tensor) -> torch.tensor:
        out = torch.zeros_like(y)
        out[:, :, :, 0] = self.t1Std*y[:, :, :, 0] + self.t1Mean
        out[:, :, :, 1] = self.t2Std*y[:, :, :, 1] + self.t2Mean
        return out
