#===============================================================================
#import packages
import argparse

import torch
import numpy as np
from os.path import dirname
from torch import nn, optim
from collections import OrderedDict
from torchvision import datasets, transforms, models





#===============================================================================
#check and pass the arguments

def pass_args():
    args = argparse.ArgumentParser(description="Here goes settings")
    args.add_argument('--arch', type=str, default='vgg16', help="specify the architecture")
    args.add_argument('--learn_rate', type=float, default=0.001, help="learning rate for the training model")
    args.add_argument('--epochs', type=int, default=5, help="epochs for training model")
    args.add_argument('--hidden_layer', type=int, default=4096, help="choose the hidden layer no")
    ar = args.parse_args()
    return ar

#===============================================================================
#define train, valid, test data sets and loader
train_transform = transforms.Compose([transforms.RandomRotation(30),
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])
valid_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])

#===============================================================================
#directories of test, train and valid
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'


#===============================================================================
#perform dataloads

# TODO: Load the datasets with ImageFolder for training set
train_datasets = datasets.ImageFolder(train_dir, transform=train_transform)

# TODO: Using the image datasets and the trainforms, define the trainloaders
trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)

valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transform)

# TODO: Using the image datasets and the trainforms, define the trainloaders
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=False)



#===============================================================================
#define the model classifier

def ourmodel(arch, hidden_layer):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)

    if arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    
    #classifier part
    model.classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(model.classifier[0].in_features, hidden_layer)),
                                ('relu1',  nn.ReLU()),
                                ('drout1', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(hidden_layer, 1000)),
                                ('relu2', nn.ReLU()),
                                ('drout2', nn.Dropout(p=0.5)),
                                ('fc3', nn.Linear(1000, 102)),
                                ('output', nn.LogSoftmax(dim=1))]))
    print(model)





#train the model and check the accuracy on the valid dataset

#save the checkpoint in the directory mentioned


#main function
def main():
    ar = pass_args()
    model = ourmodel(ar.arch, ar.hidden_layer)



#call the main function
if __name__ == '__main__' : main()
