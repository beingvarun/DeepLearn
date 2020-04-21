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
    args.add_argument('--hidden_units', type=int, default=4096, help="choose the hidden layer no")
    args.add_argument('--save_dir', help="specify the directory to save the checkpoint")
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
#check cuda is available otherwise use cuda

def availableDevice():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device










#===============================================================================
#define the model classifier

structure = {"vgg16":25088, "densenet121":1024}

def ourmodel(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)

    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

#classifier part

    model.classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(structure[arch], 4096)),
                                ('relu1',  nn.ReLU()),
                                ('drout1', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(4096, 1000)),
                                ('relu2', nn.ReLU()),
                                ('drout2', nn.Dropout(p=0.5)),
                                ('fc3', nn.Linear(1000, 102)),
                                ('output', nn.LogSoftmax(dim=1))
]))
    return model
#train the model and check the accuracy on the valid dataset



def trainModel(TrainLoader, Validloader, selectedModel, epochs, optimizer, learnRate, device):
    steps =0
    print_every = 5
    running_loss = 0
    criterion = nn.NLLLoss()
    print(device)
    for e in range(epochs):
        steps += 1
        for inputs, labels in TrainLoader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = selectedModel.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


            if (steps % print_every == 0):
                print(steps)
                accuracy = 0
                valid_loss = 0
                selectedModel.eval()
                with torch.no_grad():
                    for inputs, labels in Validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = selectedModel.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)

                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print("========================")
                print("Epochs:{}/{}".format(e+1, epochs))
                print("trainloss:", running_loss/len(trainloaders))
                print("validloss:", valid_loss/len(validloaders))
                print("Accuracy", accuracy/len(validloaders))
                running_loss = 0
                selectedModel.train()
        print("======Training model completed!!==========")
        return selectedModel


#save the checkpoint in the directory mentioned
def saveCheckpoint(trainModel, checkpointName, arch):
    trainModel.class_to_idx = train_datasets.class_to_idx
    checkpoint = {
    'input_size':structure[arch],
    'output_size':102,
    'hidden_layer_1':4096,
    'hidden_layer_2':1000,
    'state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, checkpointName+'.pth')
    print("==checkpoints created!!")

#main function
def main():
    ar = pass_args()
    Epochs = ar.epochs
    LearnRate = ar.learn_rate
    device = availableDevice()
    Model = ourmodel(ar.arch, ar.hidden_units)
    Model.to(device)
    Optimizer = optim.Adam(Model.classifier.parameters(), lr=LearnRate)
    trainedModel = trainModel(trainloaders, validloaders, Model, Epochs, Optimizer, LearnRate, device)
    saveCheckpoint(trainedModel, "checkpoint", ar.arch)


#call the main function
if __name__ == '__main__' : main()
