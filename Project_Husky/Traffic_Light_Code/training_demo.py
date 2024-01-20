#! /usr/bin/env python3
#------------------------------------------------------------------------------------------------------------------------------
#                                              University of Applied Sciences Munich
#                                              Dept of Electrical Enineering and Information Technology
#                                              Institute for Applications of Machine Learning and Intelligent Systems (IAMLIS)
#                                                                                                      (c) Alfred Schöttl 2023
#-----------------------------------------------------------------------------------------------------------------------------
# dataloader_demo: A publisher node
#------------------------------------------------------------------------------------------------------------------------------
# A little demo program to illustrate a basic trainer.
#------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data import TensorDataset
import torchvision
from torchvision.transforms import ToTensor, Compose, Lambda
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

root_dir = os.path.dirname(__file__)


batch_size = 128

#####################################################################################
# The data:
train_ds = torchvision.datasets.MNIST(root_dir+'/mnist_data', train=True, download=True, 
                                      transform=Compose([ToTensor(), Lambda(torch.flatten)]))
train_loader = dataloader.DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=0)
#####################################################################################


#####################################################################################
# The model:
def model(X, W, b):
    return torch.softmax(X@W+b, dim=-1)

def loss(Y_true, Y_pred):
    Y_true_oh = torch.nn.functional.one_hot(Y_true, num_classes=10)   # recode to a one-hot tensor
    sample_loss = -(Y_true_oh*torch.log(Y_pred+1e-7)).sum(axis=1)
    return sample_loss.mean()
####################################################################################


####################################################################################
# The Trainer
def train_step(X, Y_true, W, b, opt):
    Y_pred = model(X, W, b)                # predict
    L = loss(Y_true, Y_pred)               # compute the loss
    L.backward()                           # compute the gradients for the optimizer
    opt.step()                             # call the optimizer, modifies the weights
    opt.zero_grad()  
    return L.detach().numpy()              # we dont do further computations with L, the numpy value is sufficient for reporting
    
def train(train_ds, alpha):
    W = torch.empty(784, 10, requires_grad=True, dtype=torch.float32)         # initialize the weights
    torch.nn.init.normal_(W, std=0.001)
    b = torch.zeros(1, 10, requires_grad=True, dtype=torch.float32)
    opt = torch.optim.SGD([W, b], lr=alpha)                                   # choose the optimizer
    hist = { 'loss': [] }
    for epoch in range(2):                                                    # repeat for n epochs
        for step, (X, Y_true) in enumerate(train_loader):                     # repeat for all mini-batches
            L = train_step(X, Y_true, W, b, opt)
            hist['loss'].append(L)                                            # logging
            if step % 100 == 0:
                print(f'Epoch: {epoch}, step {step*batch_size:5d}/{len(train_loader.dataset)}:  loss: {L:.6f}')
    return W, b, hist
####################################################################################


W, b, hist = train(train_ds, alpha=0.001)

plt.plot(hist['loss'])
plt.grid()
plt.show(block=True)
print('finished.')