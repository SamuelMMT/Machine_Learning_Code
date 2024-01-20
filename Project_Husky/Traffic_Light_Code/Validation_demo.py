#! /usr/bin/env python3
#------------------------------------------------------------------------------------------------------------------------------
#                                              University of Applied Sciences Munich
#                                              Dept of Electrical Enineering and Information Technology
#                                              Institute for Applications of Machine Learning and Intelligent Systems (IAMLIS)
#                                                                                                      (c) Alfred Schöttl 2023
#-----------------------------------------------------------------------------------------------------------------------------
# cv_demo
#------------------------------------------------------------------------------------------------------------------------------
# A little demo program for convolutional nets with a training loop with validation.
#------------------------------------------------------------------------------------------------------------------------------

import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data import TensorDataset
from torch import nn
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
root_dir = os.path.dirname(__file__)

batch_size = 128

#####################################################################################
# The data:
train_ds = torchvision.datasets.CIFAR10(root_dir+'/cifar_data', train=True, download=True, 
                                      transform=ToTensor())
test_ds = torchvision.datasets.CIFAR10(root_dir+'/cifar_data', train=False, download=True, 
                                      transform=ToTensor())
train_dl = dataloader.DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True)
test_dl = dataloader.DataLoader(test_ds, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=True)
#####################################################################################


#####################################################################################
# The model:
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv2d(3, 10, 3, stride=2, padding=1)
        self.cv2 = nn.Conv2d(10, 50, 3, stride=2, padding=1)  
        self.lin1 = nn.Linear(8*8*50, 10)  
        
    def forward(self, X0):
        X1 = torch.relu(self.cv1(X0))
        X2 = torch.relu(self.cv2(X1))
        X3 = torch.flatten(X2, start_dim=1)
        X4 = torch.softmax(self.lin1(X3), dim=-1)
        return X4

    def loss(self, Y_true, Y_pred):                                       # we put the loss fct in the class, this is not required
        Y_true_oh = torch.nn.functional.one_hot(Y_true, num_classes=10)   # recode to a one-hot tensor
        sample_loss = -(Y_true_oh*torch.log(Y_pred+1e-7)).sum(axis=1)
        return sample_loss.mean()
####################################################################################


####################################################################################
# The Trainer
def train_step(X, Y_true, mdl, opt):
    Y_pred = mdl(X)                        # predict
    L = mdl.loss(Y_true, Y_pred)           # compute the loss
    L.backward()                           # compute the gradients for the optimizer
    opt.step()                             # call the optimizer, modifies the weights
    opt.zero_grad()  
    return L.detach().numpy()              # we dont do further computations with L, the numpy value is sufficient for reporting

def val_step(X, Y_true, mdl):
    Y_pred = mdl(X)                        # predict
    L = mdl.loss(Y_true, Y_pred)           # compute the loss
    return L.detach().numpy()              # we dont do further computations with L, the numpy value is sufficient for reporting    
    
def train(train_dl, mdl, alpha, n_epochs):
    opt = torch.optim.Adam(mdl.parameters(), lr=alpha)                        # choose the optimizer
    hist = { 'loss': [], 'loss_val': [], 'time_val': [] }
    k = 0
    for epoch in range(n_epochs):                                             # repeat for n epochs
        for step, (X, Y_true) in enumerate(train_dl):                         # repeat for all mini-batches
            mdl.train()
            L = train_step(X, Y_true, mdl, opt)
            hist['loss'].append(L)                                            # logging
            if k % 100 == 0:
                print(f'Epoch: {epoch}, step {step*batch_size:5d}/{len(train_dl.dataset)}:  loss: {L:.6f}')
            if k % 300 == 0 or step == len(train_dl)-1 and epoch == n_epochs-1:
                mdl.eval()
                L_val = 0.
                for step_val, (X_val, Y_val) in enumerate(test_dl):
                    L_val += val_step(X_val, Y_val, mdl)
                hist['loss_val'].append(L_val/step_val)
                hist['time_val'].append(k)
                print(f'>>>  Validation:             loss: {L_val/step_val:.6f}')
            k += 1
    return hist
####################################################################################

mdl = Model()
hist = train(train_dl, mdl, alpha=0.001, n_epochs=3)

plt.plot(hist['loss'], 'b')
plt.plot(hist['time_val'], hist['loss_val'], 'r')
plt.grid()
plt.show(block=True)

it = iter(test_dl)
X_test, Y_test = next(it)
Y_pred = mdl(X_test)
print(Y_pred.detach().numpy())