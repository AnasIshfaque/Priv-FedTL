#!/usr/bin/env python
# coding: utf-8

# # CIFAR10 Federated Mobilenet Client Side
# This code is the server part of CIFAR10 federated mobilenet for **multi** client and a server.

# In[3]:


users = 4 # number of clients


# In[4]:


import os
import h5py
import numpy as np
import subprocess
import time
import copy
import random
import shutil
import glob

import socket
import struct
import pickle

import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms,models
import torch.optim as optim
# from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from torch import tensor
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from pathlib import Path
from gpiozero import CPUTemperature
import medmnist
from medmnist import INFO, Evaluator
from datetime import datetime
from torch.utils.data import Subset

# In[2]:

dataset_name = "pneumoniamnist"
model_name = "squeezenet"

def getFreeDescription():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 1:
            return (line.split()[0:7])


def getFree():
    free = os.popen("free -h")
    i = 0
    while True:
        i = i + 1
        line = free.readline()
        if i == 2:
            return (line.split()[0:7])


def printPerformance():
    cpu = CPUTemperature()

    print("temperature: " + str(cpu.temperature))

    description = getFreeDescription()
    mem = getFree()

    print(description[0] + " : " + mem[1])
    print(description[1] + " : " + mem[2])
    print(description[2] + " : " + mem[3])
    print(description[3] + " : " + mem[4])
    print(description[4] + " : " + mem[5])
    print(description[5] + " : " + mem[6])


printPerformance()

subprocess.Popen(["./check_device.sh"])

start_time = datetime.now()
start_time = start_time.strftime("%H:%M:%S")
print("Main code exe started at ", start_time)

torch.manual_seed(1)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

client_order = int(input("client_order(start from 0): "))


num_traindata = 4708 // users # divide training instances evenly to each clients


# ## Data load

mean = np.array([0.485,0.456,0.406])
std = np.array([0.229,0.224,0.225])
data_transforms = {
  'train':transforms.Compose([
      transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean,std)
  ]),
  'test':transforms.Compose([
      transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean,std)
  ])
}

indices = list(range(4708))

lower_idx = num_traindata * client_order
upper_idx = num_traindata * (client_order + 1)

#giving the extra data instance to the last client
if (client_order+1 == users):
    upper_idx += 1
    
part_tr = indices[lower_idx : upper_idx]

# Loading MedMNIST dataset
info = INFO['pneumoniamnist']
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])
class_labels = info['label']
print(info['label'])

BATCH_SIZE = 4

sets = ['train','test']
image_datasets = {x:DataClass(split=x, transform=data_transforms[x], download=True, size=128)
                for x in ['train','test']}

trainset_sub = Subset(image_datasets['train'], part_tr)

print(f"trainset size: {len(image_datasets['train'])}, trainset_sub: {len(trainset_sub)}")
train_loader = torch.utils.data.DataLoader(trainset_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

# classes = ('Bluetooth', 'Humidity', 'Transistor')

# ### Number of total batches
train_total_batch = len(train_loader)
print(f'len(train_loader): {train_total_batch}')

sq_model = models.squeezenet1_1(weights=True)
sq_model.to(device)

#freezing previous layers
for param in sq_model.features.parameters():
    param.requires_grad = False

# modifying the last layer to match desired output class
in_ftrs = sq_model.classifier[1].in_channels
features = list(sq_model.classifier.children())[:-3] # Remove last 3 layers
features.extend([nn.Conv2d(in_ftrs, n_classes, kernel_size=1)]) # Add
features.extend([nn.ReLU(inplace=True)]) # Add
features.extend([nn.AdaptiveAvgPool2d(output_size=(1,1))]) # Add
sq_model.classifier = nn.Sequential(*features)

local_weights = copy.deepcopy(sq_model.state_dict())
# In[16]:


lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(sq_model.parameters(), lr=lr, momentum=0.9)

rounds = 10 # default
local_epochs = 1 # default


# ## Socket initialization
# ### Required socket functions

# In[17]:


def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    # encrypt msg
    sock.sendall(msg)

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        # decrypt
        if not packet:
            return None
        data += packet
    return data


printPerformance()


# ### Set host address and port number
host = input("IP address: ")
port = 10080
max_recv = 100000


# ### Open the client socket
s = socket.socket()
s.connect((host, port))


# ## SET TIMER
start_time = time.time()    # store start time
print("timmer start!")


msg = recv_msg(s)
rounds = msg['rounds'] 
client_id = msg['client_id']
local_epochs = msg['local_epoch']
send_msg(s, len(trainset_sub))


# In[22]:


# update weights from server
# train
for r in range(rounds):  # loop over the dataset multiple times
    
    last_layer_list = recv_msg(s)
    # Updating the global weight's last layer
    local_weights['classifier.1.weight'] = last_layer_list[0]
    local_weights['classifier.1.bias'] = last_layer_list[1]
    sq_model.load_state_dict(local_weights)
    sq_model.train()
    for local_epoch in range(local_epochs):
        
        for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Round '+str(r+1)+'_'+str(local_epoch+1))):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # if len(labels) == BATCH_SIZE:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = sq_model(inputs)
            _,preds = torch.max(outputs,1)
            labels = labels.squeeze().long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    msg = [sq_model.state_dict()['classifier.1.weight'], sq_model.state_dict()['classifier.1.bias']]
    # msg = mobile_net.state_dict()
    send_msg(s, msg)

print('Finished Training')

printPerformance()

end_time = time.time()  #store end time
print("Training Time: {} sec".format(end_time - start_time))

