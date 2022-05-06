#!/usr/bin/env python3
"""
template.py: Describe what template does.

Created on Tue Apr 19 16:38:41 2022
@author: Sam Adams
"""

from datetime import date
import math
import os
os.chdir('D:/ECE6960_FinalProject/')

#%% Define parameters/constants
trainSets = ['./Datasets/20220323dess_1.mat',
             './Datasets/20220323dess_2.mat',
             './Datasets/20220323dess_3.mat',
             './Datasets/20220323dess_4.mat',
             './Datasets/20220323dess_5.mat',
             './Datasets/20220323dess_6.mat',
             './Datasets/20220323dess_7.mat',
             './Datasets/20220323dess_8.mat',
             './Datasets/20220323dess_9.mat',
             './Datasets/20220323dess_10.mat',
             './Datasets/20220323dess_11.mat']
strdFile = 'standardization.json'

# Data parameters
batchSize = 5
trainShape = 128

# Network parameters
convChannels = 32
convDepth, encodeDepth = 2, 4

# Training parameters
lr = 1e-3
numEpochs = 50
printEvery = max(1, math.floor(numEpochs/20))

# Load/save datasets
loadParams = None #"2022-04-21_convNet_convChannels=32_convDepth=2_encodeDepth=4.pt"
saveParams = f"{date.today().isoformat()}_convNet_convChannels={convChannels}_convDepth={convDepth}_encodeDepth={encodeDepth}.pt"

#%% Create datasets
from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import datautils

# Create standardization object
strd = datautils.Standardizer(strdFile)

dess = torch.empty((192, 192, 5*len(trainSets), 6, 2), dtype=torch.complex64)
t1map = torch.empty((192, 192, 5*len(trainSets), 1))
t2map = torch.empty((192, 192, 5*len(trainSets), 1))
mask = np.empty((192, 192, 5*len(trainSets), 1))

for idx, fpath in enumerate(trainSets):
    matfile = loadmat(fpath)
    
    dess[:, :, (idx*5):((idx+1)*5), :, :] = torch.tensor(matfile['dess'])
    t1map[:, :, (idx*5):((idx+1)*5), 0] = torch.tensor(matfile['t1map'])
    t2map[:, :, (idx*5):((idx+1)*5), 0] = torch.tensor(matfile['t2map'])
    mask[:, :, (idx*5):((idx+1)*5), 0] = np.array(matfile['mask'])

dess = strd.standardize_features(dess)
tmp = strd.standardize_targets(torch.cat((t1map, t2map), dim=3))
t1map, t2map = tmp[:, :, :, 0], tmp[:, :, :, 1]

testIdx = np.arange(3, dess.shape[2], 5)
trainSlices = np.arange(dess.shape[2])
testSlices = trainSlices[testIdx]
trainSlices = np.delete(trainSlices, testIdx)

# Create training dataset
trainData = datautils.DESSImageData(torch.abs(dess[:, :, trainSlices]), torch.angle(dess[:, :, trainSlices]), t1map[:, :, trainSlices], t2map[:, :, trainSlices])
trainLoader = torch.utils.data.DataLoader(trainData, batch_size=batchSize, shuffle=True)
testData = datautils.DESSImageData(torch.abs(dess[:, :, testSlices]), torch.angle(dess[:, :, testSlices]), t1map[:, :, testSlices], t2map[:, :, testSlices])
testLoader = torch.utils.data.DataLoader(testData, batch_size=batchSize, shuffle=True)

# Establish data augmentation transforms
trnsfrm = nn.Sequential(
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomCrop(trainShape))

#%% Train network
import torch
import torch.nn as nn
import nnutils

genNet = nnutils.GeneratorNet(featureChannels=24, targetChannels=2,
                              convChannels=convChannels,
                              encodeDepth=encodeDepth, blockDepth=convDepth)
if loadParams is not None:
    genNet.load_state_dict(torch.load(loadParams))
discNet = nnutils.DiscriminatorNet(targetChannels=2, convChannels=16,
                                   encodeDepth=5, blockDepth=2)

# Send networks to GPU, if available
genNet.to(nnutils.try_gpu()), discNet.to(nnutils.try_gpu())

loss = nn.L1Loss(reduction='none')
genOptim = torch.optim.Adam(genNet.parameters(), lr=lr)
discOptim = torch.optim.SGD(discNet.parameters(), lr=lr, momentum=0.5, weight_decay=1e-4)

trainLosses, trainMSEs,\
    testLosses, testMSEs =\
    nnutils.train_SegAN(genNet, genOptim, discNet, discOptim,
                        trainLoader, testLoader, numEpochs,
                        loss=loss, printEvery=printEvery, savePath=saveParams)

print("")
print("Training complete!")

# Bring networks back to CPU
genNet.to(torch.device(type='cpu')), discNet.to(torch.device(type='cpu'))
torch.cuda.empty_cache()

#%% Summarize Results
import matplotlib.pyplot as plt
import evalutils

### Plot performance during training
plt.figure(figsize=(8, 3.5))

# Plot multiscale loss performance
plt.subplot(1, 2, 1)
plt.semilogy(np.arange(numEpochs) + 1, trainLosses)
plt.semilogy(np.arange(numEpochs) + 1, testLosses)
plt.title("Multi-scale Adversarial Loss")
plt.xlabel("Epoch number")
plt.ylabel("MSAE performance")
plt.legend(("Training", "Testing"))

# Plot loss
plt.subplot(1, 2, 2)
plt.semilogy(np.arange(numEpochs) + 1, trainMSEs)
plt.semilogy(np.arange(numEpochs) + 1, testMSEs)
plt.title("Loss")
plt.xlabel("Epoch number")
plt.ylabel("Loss performance")
plt.legend(("Training", "Testing"))

plt.suptitle(saveParams)
plt.tight_layout()
plt.show()

print("* Note that training curves reflect performance with dropout!")

#%% Evaluate network

# Load the saved network
genNet.load_state_dict(torch.load('2022-04-26_convNet_convChannels=32_convDepth=2_encodeDepth=4.pt'))

genNet.eval(); discNet.eval()

x, yData = iter(testLoader).next()
yData = strd.destandardize_targets(torch.movedim(yData, (0, 1), (2, 3)))

yGen = genNet(x)
yGen = strd.destandardize_targets(torch.movedim(yGen, (0, 1), (2, 3)))

evalutils.slice_results(0, yData[:, :, :, 0].detach().numpy(),
                        yData[:, :, :, 1].detach().numpy(),
                        yGen[:, :, :, 0].detach().numpy(),
                        yGen[:, :, :, 1].detach().numpy())

# Print performance
print("")
print(" Dataset |   MSAL  |  Loss")
print("----------------------------")
tmp = nnutils.evaluate_performance_SegAN(genNet, discNet, trainLoader, loss)
print(f"  Train. | {tmp[0]:3.1e} |  {tmp[1]:3.1e}")
tmp = nnutils.evaluate_performance_SegAN(genNet, discNet, testLoader, loss)
print(f" Testing | {tmp[0]:3.1e} |  {tmp[1]:3.1e}")

t1Data, t1Gen = yData[:, :, :, 0], yGen[:, :, :, 0]
t2Data, t2Gen = yData[:, :, :, 1], yGen[:, :, :, 1]

evalutils.plot_bland_altman(t1Data[np.nonzero(mask[:, :, testSlices[0]])].flatten().detach().numpy(),
                            t1Gen[np.nonzero(mask[:, :, testSlices[0]])].flatten().detach().numpy(),
                            'STIR T1', 'Neural T1', 'T1', units='ms', rng=(0, 2000))
evalutils.plot_bland_altman(t2Data[np.nonzero(mask[:, :, testSlices[0]])].flatten().detach().numpy(),
                            t2Gen[np.nonzero(mask[:, :, testSlices[0]])].flatten().detach().numpy(),
                            'SE-ME T2', 'Neural T2', 'T2', units='ms', rng=(0, 250))

#%% Summarize network structure
import math

print(genNet)
print("~ Generator parameters ~")
paramTot = 0
for name, param in genNet.named_parameters():
    numParams = math.prod(param.shape)
    print(f"{numParams:9d} parameters in {name}")
    paramTot += numParams
print( "---------------------")
print(f"{paramTot:9d} parameters total")

print("")
print(discNet)
print("~ Discriminator parameters ~")
paramTot = 0
for name, param in discNet.named_parameters():
    numParams = math.prod(param.shape)
    print(f"{numParams:9d} parameters in {name}")
    paramTot += numParams
print( "---------------------")
print(f"{paramTot:9d} parameters total")
