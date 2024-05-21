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
from torch.optim import lr_scheduler
from torch import tensor
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from pathlib import Path
from gpiozero import CPUTemperature
import medmnist
from medmnist import INFO, Evaluator
from datetime import datetime
from torch.utils.data import Subset

import tenseal as ts

# In[2]:

dataset_name = "bloodmnist"
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

process = subprocess.Popen(["../../playground/Developing Transfer Learning Model/check_device.sh"])

start_time = datetime.now()
start_time = start_time.strftime("%H:%M:%S")
print("Main code exe started at ", start_time)

torch.manual_seed(42)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

client_order = int(input("client_order(start from 0): "))


num_traindata = 11959 // users # divide training instances evenly to each clients


# ## Data load

mean = np.array([0.485,0.456,0.406])
std = np.array([0.229,0.224,0.225])
data_transforms = {
  'train':transforms.Compose([
      # transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean,std)
  ]),
  'test':transforms.Compose([
      # transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean,std)
  ])
}

indices = list(range(11959))

lower_idx = num_traindata * client_order
upper_idx = num_traindata * (client_order + 1)

#giving the extra data instance to the last client
if (client_order+1 == users):
    upper_idx += 1
    
part_tr = indices[lower_idx : upper_idx]

# Loading MedMNIST dataset
info = INFO[dataset_name]
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


# loading the TenSEAL context
with open('../../playground/Developing H.E for FL/shared_context.pkl', 'rb') as inp:
    shared_context_bin = pickle.load(inp)

shared_context = ts.context_from(shared_context_bin)
sk = shared_context.secret_key()


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
optimizer = optim.SGD(sq_model.parameters(), lr=lr)
step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

rounds = 10 # default
local_epochs = 1 # default

total_encrypt_time = 0
total_decrypt_time = 0

# ## Socket initialization
# ### Required socket functions

# In[17]:


def send_msg(sock, msg, encrypt=True):
    # prefix each message with a 4-byte length in network byte order
    global total_encrypt_time
    if encrypt:
        encrypt_start = time.time()
        plain_ten = ts.plain_tensor(msg)
        encrypted_ten = ts.ckks_tensor(shared_context, plain_ten)
        msg = encrypted_ten.serialize()
        encrypt_end = time.time()
        total_encrypt_time += (encrypt_end - encrypt_start)
    else:        
        msg = pickle.dumps(msg)
        
    msg = struct.pack('>I', len(msg)) + msg
    # encrypt msg
    sock.sendall(msg)

def recv_msg(sock, decrypt=True):
    # read message length and unpack it into an integer
    global total_decrypt_time
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    if decrypt:
        decrypt_start = time.time()
        msg = ts.ckks_tensor_from(shared_context, msg)
        msg = torch.tensor(msg.decrypt(sk).tolist())
        decrypt_end = time.time()
        total_decrypt_time += (decrypt_end - decrypt_start)
    else:
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


msg = recv_msg(s, False)
rounds = msg['rounds'] 
client_id = msg['client_id']
local_epochs = msg['local_epoch']
last_layer_list_len = msg['last_layer_list_len']
send_msg(s, len(trainset_sub), False)


# In[22]:


# update weights from server
# train

for r in range(rounds):  # loop over the dataset multiple times
    print(f'recieving global weights in round {r}...')
    if r == 0:
        last_layer_list = recv_msg(s, False) # first round recieve plain list
    else:
        last_layer_list = []
        for i in range(last_layer_list_len):
            param = recv_msg(s)
            last_layer_list.append(param)
    print(f'global weights recieved in round {r}!')
    # Updating the global weight's last layer
    local_weights['classifier.1.weight'] = last_layer_list[0]
    local_weights['classifier.1.bias'] = last_layer_list[1]
    sq_model.load_state_dict(local_weights)
    sq_model.train()
    for local_epoch in range(local_epochs):
        
        for i, data in enumerate(tqdm(train_loader, ncols=100, desc='Round '+str(r+1)+'_'+str(local_epoch+1))):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
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
    print(f'sending encrypted local weights in round {r}...')
    for param in msg:    
        send_msg(s, param) # send encrypted weights one tensor at a time
    print(f'local weights sent in round {r}!')

print('Finished Training')

printPerformance()

end_time = time.time()  #store end time
print("Training Time: {} sec".format(end_time - start_time))
print(f'Total encryption time: {total_encrypt_time}')
print(f'Total decryption time: {total_decrypt_time}')

process.kill()
process.wait()
