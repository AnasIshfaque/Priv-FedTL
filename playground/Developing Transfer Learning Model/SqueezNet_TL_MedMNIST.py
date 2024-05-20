import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import tensor
import numpy as np
import torchvision
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time
import os
import subprocess
import copy
import random
import shutil
import glob
import torchmetrics
from pathlib import Path
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
# from sklearn.model_selection import StratifiedKFold
from gpiozero import CPUTemperature
import medmnist
from medmnist import INFO, Evaluator
from datetime import datetime
import random

torch.manual_seed(42)

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

def print_metrics(metrics):
    #total_time = metrics["time"]*5
    print(f'Average accuracy: {metrics["accuracy"]:.4f}')
    print(f'Average precision: {metrics["precision"]:.4f}')
    print(f'Average recall: {metrics["recall"]:.4f}')
    print(f'Average F1 score: {metrics["f1"]:.4f}')
    # print(f'Average Time elapsed: {metrics["time"]:.4f} seconds')

printPerformance()

subprocess.Popen(["./check_device.sh"])

start_time = datetime.now()
start_time = start_time.strftime("%H:%M:%S")
print("Main code exe started at ", start_time)

torch.manual_seed(42)
device = "cpu"

def get_transforms():
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
    return data_transforms
def train_model(model, dataset_name, criterion, optimizer, scheduler,  dataloaders, dataset_sizes, num_classes=3, num_epochs=10):
    since = time.time()
    # torch.cuda.reset_peak_memory_stats(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # create a list to store the training and validation accuracy values
    train_acc_list = []
    val_acc_list = []

    # create a list to store the training and validation loss values
    train_loss_list = []
    val_loss_list = []

    # initialize metric
    metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
    predicted_labels = []
    ground_truth_labels = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-'*10)

        #Training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            #Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #forward
                #track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #_,preds = torch.max(outputs,1)
                    #loss = criterion(outputs,labels)
                    #googlenetcprfix
                    if str(type(outputs)) == "<class 'torch.Tensor'>":
                        _,preds = torch.max(outputs,1)
                        labels = labels.squeeze().long()
                        loss = criterion(outputs,labels)
                    else :
                        _,preds = torch.max(outputs.logits,1)
                        loss = criterion(outputs.logits,labels)

                    #backward + optimize only in train
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                #statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_acc_list.append(epoch_acc.item())
                train_loss_list.append(epoch_loss)
            elif phase == 'test':
                val_acc_list.append(epoch_acc.item())
                val_loss_list.append(epoch_loss)

            predicted_labels.append(preds.cpu())
            ground_truth_labels.append(labels.cpu())


            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            #deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    #calculate accuracy
    predicted_labels = torch.cat(predicted_labels)
    ground_truth_labels = torch.cat(ground_truth_labels)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    accuracy(predicted_labels, ground_truth_labels)
    print(f'Accuracy: {accuracy.compute():.4f}')

    #calculate precision
    precision = Precision(task="multiclass", average='macro', num_classes=num_classes)
    precision(predicted_labels, ground_truth_labels)
    print(f'Precision: {precision.compute():.4f}')

    #calculate recall
    recall = Recall(task="multiclass", average='macro', num_classes=num_classes)
    recall(predicted_labels, ground_truth_labels)
    print(f'Recall: {recall.compute():.4f}')

    #calculate f1 score
    f1 = F1Score(task="multiclass", average='macro', num_classes=num_classes)
    f1(predicted_labels, ground_truth_labels)
    print(f'F1: {f1.compute():.4f}')

    #calculate confusion matrix
    cm = torchmetrics.functional.confusion_matrix(predicted_labels, ground_truth_labels, num_classes=num_classes, task="multiclass")
    print(f'Confusion Matrix: \n{cm}')

    #plot the training and validation accuracy
    plt_dir = './plots'
    
    plt.figure(figsize=(10, 6))

    plt.plot(train_acc_list, label='Training Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(plt_dir+'/accuracy_curves/'+ f'acc_{dataset_name}_{model.__class__.__name__}.png')

    #plot the training and validation loss

    plt.figure(figsize=(10, 6))

    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(plt_dir+'/loss_curves/'+ f'loss_{dataset_name}_{model.__class__.__name__}.png')

    metrics = {}
    metrics['accuracy'] = accuracy.compute()
    metrics['precision'] = precision.compute()
    metrics['recall'] = recall.compute()
    metrics['f1'] = f1.compute()


    #load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics

def fineTune(model_name, dataset_name):
    info = INFO[dataset_name]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    DataClass = getattr(medmnist, info['python_class'])
    class_labels = info['label']
    print(info['label'])

    data_transforms = get_transforms()

    BATCH_SIZE = 4

    sets = ['train','test']
    image_datasets = {x:DataClass(split=x, transform=data_transforms[x], download=True, size=128)
                    for x in ['train','test']}
    
    # using only a part of the training data to make fair comparision with FL systems
    num_data_parts = 4
    num_traindata = 4708 // num_data_parts

    indices = list(range(4708))

    client_order = random.randint(0, num_data_parts-1) # randomly choosing a part
    lower_idx = num_traindata * client_order
    upper_idx = num_traindata * (client_order + 1)
    
    #giving the extra data instance to the last client
    if (client_order+1 == num_data_parts):
        upper_idx += 1
        
    part_tr = indices[lower_idx : upper_idx]

    trainset_sub = Subset(image_datasets['train'], part_tr)
    
    dataloaders = {'train': torch.utils.data.DataLoader(trainset_sub,batch_size=BATCH_SIZE,
                                                shuffle=True,num_workers=0),
                   'test': torch.utils.data.DataLoader(image_datasets['test'],batch_size=2*BATCH_SIZE,
                                                shuffle=True,num_workers=0)
                  }

    # dataset_sizes = {x:len(image_datasets[x]) for x in ['train','test']}
    dataset_sizes = {'train': len(trainset_sub),
                     'test': len(image_datasets['test'])}
    
    print(dataset_sizes['train'])
    
    pretrained_model = models.squeezenet1_1(pretrained=True)
    
    #freezing previous layers
    for param in pretrained_model.features.parameters():
        param.requires_grad = False

    if model_name == 'squeezenet':
        # Newly created modules have require_grad=True by default
        num_features = pretrained_model.classifier[1].in_channels
        features = list(pretrained_model.classifier.children())[:-3] # Remove last 3 layers
        features.extend([nn.Conv2d(num_features, n_classes, kernel_size=1)]) # Add
        features.extend([nn.ReLU(inplace=True)]) # Add
        features.extend([nn.AdaptiveAvgPool2d(output_size=(1,1))]) # Add
        pretrained_model.classifier = nn.Sequential(*features)
    else:
        print("Invalid model name")

    pretrained_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(pretrained_model.parameters(),lr=0.01)
    
    #scheduler
    step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)
    
    model_ft, metrics = train_model(pretrained_model, dataset_name, criterion, optimizer, step_lr_scheduler, 
                                    dataloaders, dataset_sizes, n_classes, num_epochs=10)

    
    #save the best model
    
    # 1. Create models directory
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    # 2. Create model save path
    MODEL_NAME = dataset_name+"_"+model_name+"_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    
    # 3. Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_ft.state_dict(),
               f=MODEL_SAVE_PATH)




# finetune the model
fineTune(model_name, dataset_name)

process.kill()
process.wait()

printPerformance()
