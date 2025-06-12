from tqdm.autonotebook import tqdm
import os,sys,humanize,psutil,GPUtil
import time
import math
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from Attention import RNN_Dataset, SmarterAttentionNet, train_network_reg, graph_results, return_true_pred, train_network_reg_mixup
from util import rsnet18
device = torch.device("cuda")
Qc_Dataset = torch.load('Qc.pt')
# Qc_Dataset = torch.load('Qc_3680_230.pt')
# variables
eta = 0.001            # learning rate
step_size = 10         # Period of learning rate decay, see torch.optim.lr_scheduler.StepLR
gamma = 0.5            # Multiplicative factor of learning rate decay. Default: 0.1
epoch_no = 1    # number of epochs that will be used in training
activation_f='ReLU'
att_active_f='ReLU'
optimizer_f = torch.optim.AdamW

n_freq_training = 400   # number of frequencies will be used for training
n_gap = 40           # number of frequencies between training and testing datasets
input_dim=25     # attention mechanism input dimension
num_neurons=256  # number of neurons to be used in the NN
n_test = 1000     # testing dataset size
num_layers= 5
# features (we can't change these)
n_feature = 4  # number of features describing the ring resonator
nf = 601         # number of Qc values for each device
fmax = 500       # max sim. frequency THz
df = 0.5         # sim frequeny difference
output_dim=nf-n_freq_training-n_gap
T_length=n_feature+n_freq_training//input_dim
fmin_test = fmax-df*(Qc_Dataset.shape[1]-n_feature-n_freq_training-n_gap)
n_training_feature = n_feature+n_freq_training
freqs = np.arange(fmin_test+df, fmax+df, df)
# dataset_instance = RNN_Dataset(dataset=Qc_Dataset, n=n_freq_training, input_dim=input_dim, ngap=n_gap)
# train_indices = list(range(0, Qc_Dataset.shape[0]-1000))
# test_indices = list(range(Qc_Dataset.shape[0]-1000, Qc_Dataset.shape[0]))
# train_data = Subset(dataset_instance, train_indices)
# test_data = Subset(dataset_instance, test_indices)
train_data, test_data = torch.utils.data.random_split(RNN_Dataset(dataset=Qc_Dataset,n=n_freq_training,input_dim=input_dim, ngap = n_gap), (Qc_Dataset.shape[0]-1000, 1000))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
# model = SmarterAttentionNet(T_length=T_length, input_dim=input_dim, num_neurons=num_neurons, output_dim=output_dim, activation=activation_f , att_active=att_active_f)
model = rsnet18(num_classes=output_dim)
# loss_func = nn.MSELoss()
# model = NN()
# model = recurrent_model(T_length=T_length, input_dim=input_dim,  num_layers=num_layers, num_neurons=num_neurons, output_dim=output_dim, model_type='RNN', bidirectional=False)
print(model)
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)

loss_func = RMSELoss()
optimizer = optimizer_f(model.parameters(), lr=eta)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start = time.time()
results = train_network_reg(model, loss_func, train_loader, test_loader=test_loader, epochs=epoch_no,
                                      score_funcs={'R^2 score': r2_score}, device=device, optimizer=optimizer, lr_schedule=scheduler)
print(results)
print(results.iloc[:, -2].max())
print(results.iloc[:, -1].max())
stop = time.time()
print('Training time: %s sec' %(stop-start))
sns.lineplot(x='epoch', y='test R^2 score', data=results[1:])
plt.title('Test R^2 Score of the Additive Attention')
plt.gcf().set_size_inches(10, 6)
plt.show()
# for i in range(1,n_test,499):
#     graph_results(model, test_data,i,freqs)
# df_true, df_pred = return_true_pred(model, test_data, 500, freqs)
# df_true.to_csv('true_values.csv', index=False)
# df_pred.to_csv('predicted_values.csv', index=False)
