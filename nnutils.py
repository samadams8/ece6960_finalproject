#!/usr/bin/env python3
"""
nnutils.py: Describe what template does.

Created on Tue Apr  5 20:12:23 2022
@author: Sam Adams
"""

import math
import torch
import torch.nn as nn

def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class GeneratorNet(nn.Module):
    """U-Net for contrast translation"""
    def __init__(self, featureChannels: int, targetChannels: int, convChannels: int, encodeDepth: int=4, blockDepth: int=2):
        super(GeneratorNet, self).__init__()
        
        padding, paddingMode = 'same', 'reflect'
        kernelSize = 3
        
        # Calculate number of channes in each layer of the encoder
        encoderChannels = convChannels*2**(torch.arange(0, encodeDepth))
        decoderChannels = torch.flip(encoderChannels, (0, ))        
        
        # Prepare to iteratively define encoders/decoders
        downsamplers, encoders = [], []
        encoderChannels = encoderChannels.tolist()
        upsamplers, decoders = [], []
        decoderChannels = decoderChannels.tolist()
        
        for n in range(encodeDepth):
            if n + 1 >= encodeDepth:
                break
            
            ### Encoder branch
            # Downsample
            downsamplers.append(
                nn.Conv2d(encoderChannels[n], encoderChannels[n+1],
                          kernel_size=2, stride=2))
            # Encode
            encoders.append(Conv2dBlock(encoderChannels[n+1],
                                        encoderChannels[n+1],
                                        kernelSize, blockDepth,
                                        padding, paddingMode))
            
            ### Decoder branch
            # Upsample
            upsamplers.append(
                nn.ConvTranspose2d(decoderChannels[n], decoderChannels[n+1],
                                   kernel_size=2, stride=2))
            # Decode
            decoders.append(Conv2dBlock(decoderChannels[n],
                                        decoderChannels[n+1],
                                        kernelSize, blockDepth,
                                        padding, paddingMode))
        
        self.input = Conv2dBlock(featureChannels, encoderChannels[0],
                                  kernelSize, blockDepth,
                                  padding, paddingMode)
        self.downsamplers = nn.ModuleList(downsamplers)
        self.encoders = nn.ModuleList(encoders)
        self.upsamplers = nn.ModuleList(upsamplers)
        self.decoders = nn.ModuleList(decoders)
        self.output = nn.Conv2d(convChannels, targetChannels, kernelSize,
                                padding=padding, padding_mode=paddingMode)
    
    def forward(self, x):
        yHat = self.input(x)
        
        ### Run the encoder branch and save each output
        encodeOut = []
        for (downsampler, encoder) in zip(self.downsamplers, self.encoders):
            encodeOut.append(yHat)
            yHat = encoder(downsampler(yHat))
        
        # Reorder encode outputs to match order they will be decoded
        encodeOut.reverse()
        
        ### Run the decoder branch
        for (encode, upsampler, decoder) in zip(encodeOut, self.upsamplers, self.decoders):
            yHat = decoder(torch.cat((upsampler(yHat), encode), dim=1))
        
        yHat = self.output(yHat)
        
        return yHat

class DiscriminatorNet(nn.Module):
    """Convolutional multiscale loss network"""
    def __init__(self, targetChannels: int, convChannels: int, encodeDepth: int=4, blockDepth: int=2):
        super(DiscriminatorNet, self).__init__()
        padding, paddingMode = 'same', 'reflect'
        kernelSize = 3
        
        # Calculate number of channes in each layer of the encoder
        encoderChannels = convChannels*2**(torch.arange(0, encodeDepth))      

        # Prepare to iteratively define encoders/decoders
        downsamplers, encoders = [], []
        encoderChannels = encoderChannels.tolist()
        
        for n in range(encodeDepth):
            if n + 1 >= encodeDepth:
                break
            # Downsample
            downsamplers.append(
                nn.Conv2d(encoderChannels[n], encoderChannels[n+1],
                          kernel_size=2, stride=2))
            # Encode
            encoders.append(Conv2dBlock(encoderChannels[n+1],
                                        encoderChannels[n+1],
                                        kernelSize, blockDepth,
                                        padding, paddingMode))
        
        self.input = Conv2dBlock(targetChannels, encoderChannels[0],
                                  kernelSize, blockDepth,
                                  padding, paddingMode)
        self.downsamplers = nn.ModuleList(downsamplers)
        self.encoders = nn.ModuleList(encoders)
        
    def forward(self, x):
        y = self.input(x)
        
        ### Run the encoder branch and save each output
        l = y.flatten()
        for (downsampler, encoder) in zip(self.downsamplers, self.encoders):
            y = encoder(downsampler(y))
            l = torch.cat((l, y.flatten()))
            
        return l
    
    def multiscale_adversarial_loss(self, lossFcn, yGen, yData):
        # Evaluate discriminator on real and generated data, calculate loss
        discData, discGen = self(yData), self(yGen)
        loss = lossFcn(discData, discGen)
        
        return loss

class Conv2dBlock(nn.Module):
    def __init__(self, chIn: int, chOut: int, kernelSize: int, depth: int, padding='same', paddingMode='reflect'):
        super(Conv2dBlock, self).__init__()
        layers = []
        
        for n in range(depth):
            if n == 0:
                layers.append(nn.Conv2d(chIn, chOut, kernelSize, padding=padding, padding_mode=paddingMode))
            else:
                layers.append(nn.Conv2d(chOut, chOut, kernelSize, padding=padding, padding_mode=paddingMode))
            layers.append(nn.ReLU())
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        
        return y

def train_SegAN(genNet, genOptim, discNet, discOptim,\
                trainIter, testIter, numEpochs,\
                loss, printEvery=5, savePath=None,\
                lrScheduling=True, lrSchedPatience=5):
    trainMSALs, testMSALs = [], []
    trainLosses, testLosses = [], []
    
    if lrScheduling:
        genSched = torch.optim.lr_scheduler.ReduceLROnPlateau(genOptim, patience=lrSchedPatience, cooldown=lrSchedPatience)
        discSched = torch.optim.lr_scheduler.ReduceLROnPlateau(discOptim, patience=lrSchedPatience, cooldown=lrSchedPatience)
    
    print( "       |     Training     |     Testing")
    print( " Epoch |   MSAL    Loss   |   MSAL    Loss")
    print( "----------------------------------------------")
    testLoss, testLoss = evaluate_performance_SegAN(genNet, discNet, testIter, loss)
    print(f"     0 | -------  ------- | {testLoss:3.1e}  {testLoss:3.1e}")
    
    trainLow, testLow = math.inf, testLoss
    for epoch in range(numEpochs):
        trainMSAL, trainLoss = train_epoch_SegAN(genNet, genOptim, discNet, discOptim, trainIter, loss)
        testMSAL, testLoss = evaluate_performance_SegAN(genNet, discNet, testIter, loss)
        
        if (epoch + 1) % printEvery == 0:
            print(f"  {epoch + 1:4d} | {trainMSAL:3.1e}  {trainLoss:3.1e} | {testMSAL:3.1e}  {testLoss:3.1e}")
            
        if (savePath is not None)\
        and (trainLoss < trainLow)\
        and (testLoss < testLow):
            # Record new low values
            trainLow, testLow = trainLoss, testLoss
            # Save the network parameters
            torch.save(genNet.state_dict(), savePath)
        
        if lrScheduling:
            genSched.step(trainLoss)
            discSched.step(trainLoss)
        
        trainMSALs.append(trainMSAL); testMSALs.append(testMSAL)
        trainLosses.append(trainLoss); testLosses.append(testLoss)
    
    print(f"   end | {trainMSAL:3.1e}  {trainLoss:3.1e} | {testMSAL:3.1e}  {testLoss:3.1e}")
    
    return trainMSALs, trainLosses, testMSALs, testLosses
        
def train_epoch_SegAN(genNet, genOptim, discNet, discOptim,\
                      trainIter, loss):
    msalTot, lossTot, sampTot = 0., 0., 0.
    
    # Get device
    device = next(iter(genNet.parameters())).device
    
    ### Iterate over full training set
    for x, yData in trainIter:
        
        ## Train the discriminator up to msalThresh
        discNet.train(); genNet.eval()
        for xk, ykData in trainIter:
            # Send to same device as model
            xk, ykData = xk.to(device), ykData.to(device)
            
            ykGen = genNet(xk)
            msal = discNet.multiscale_adversarial_loss(loss, ykGen, ykData)
            
            # Step the optimizer to update the discriminator
            discOptim.zero_grad()
            (-msal).mean().backward() # ASCEND gradient (INCREASE difference between two sources)
            discOptim.step()
        
        ## Train the generator using the updated discriminator
        genNet.train(); discNet.eval()
        
        # Send to same device as model
        x, yData = x.to(device), yData.to(device)
        
        yGen = genNet(x)
        msal = discNet.multiscale_adversarial_loss(loss, yGen, yData)
        
        # Step the optimizer to update the generator
        genOptim.zero_grad()
        msal.mean().backward() # DESCEND gradient (DECREASE difference between two sources)
        genOptim.step()
        
        ### Update metric totals
        batchSize = float(x.shape[0])
        msalTot += float(msal.mean())*batchSize
        # Calculate MSE between ground truth and generated maps
        l = loss(yGen, yData)
        lossTot += float(l.mean())*batchSize
        sampTot += batchSize
    
    return msalTot/sampTot, lossTot/sampTot

def evaluate_performance_SegAN(genNet, discNet, dataIter, loss):
    msalTot, lossTot, sampTot = 0., 0., 0.
    genNet.eval(); discNet.eval()
    
    # Get device
    device = next(iter(genNet.parameters())).device
    
    with torch.no_grad():
        for x, yData in dataIter:
            # Send to same device as model
            x, yData = x.to(device), yData.to(device)
            
            yGen = genNet(x)
            msal = discNet.multiscale_adversarial_loss(loss, yGen, yData)
            
            ### Update metric totals
            batchSize = float(x.shape[0])
            msalTot += float(msal.mean())*batchSize
            # Calculate MSE between ground truth and generated maps
            l = loss(yGen, yData)
            lossTot += float(l.mean())*batchSize
            sampTot += batchSize
    
    return msalTot/sampTot, lossTot/sampTot
