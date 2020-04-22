#===============================================================================
#import packages
import argparse

import torch
import numpy as np
from os.path import dirname
from PIL import Image
from torch import nn, optim
from collections import OrderedDict
from torchvision import datasets, transforms, models





#===============================================================================
#check and pass the arguments

def pass_args():
    args = argparse.ArgumentParser(description="Here goes settings")
    args.add_argument('--filepath', type='str', help='specify the filepath')
    args.add_argument('--image_path', type='str', help='specify the image location')
    args.add_argument('--arch', type=str, default='vgg16', help="specify the architecture")
    args.add_argument('--learn_rate', type=float, default=0.001, help="learning rate for the training model")
    args.add_argument('--epochs', type=int, default=5, help="epochs for training model")
    args.add_argument('--hidden_units', type=int, default=4096, help="choose the hidden layer no")
    args.add_argument("--gpu", help="specify gpu or cpu for running the program")
    ar = args.parse_args()
    return ar


def availableDevice(gpu):
    if(gpu == None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    return device






#===============================================================================
#load checkpoint and return the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model_input = checkpoint['input_size']
    model_output = checkpoint['output_size']
    model_hidden_1 = checkpoint['hideen_layer_1']
    model_hidden_2 = checkpoint['hidden_layer_2']

    stored_architecture = checkpoint['architecture']

    if  stored_architecture=='vgg16':
        model = models.vgg16(pretrained=True)
    elif stored_architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("something went wrong!! architecture information is not available")
    model.classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(model_input, model_hidden_1)),
                                ('relu1',  nn.ReLU()),
                                ('drout1', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(model_hidden_1, model_hidden_2)),
                                ('relu2', nn.ReLU()),
                                ('drout2', nn.Dropout(p=0.5)),
                                ('fc3', nn.Linear(model_hidden_2, model_output)),
                                ('output', nn.LogSoftmax(dim=1))
    ]))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model

#===============================================================================
#processing image input
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    img = Image.open(image)
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    transformed_image = transform(img)
    return transformed_image
#===============================================================================
#show the image selected
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

#===============================================================================
#predict the image type
def predict(image_path, model, topk=5):
    image_data = process_image(image_path)
    image_data = image_data.unsqueeze_(0)
    image_data = image_data.to(device)
    with torch.no_grad():
        output = model.forward(image_data)

    ps = torch.exp(output)
    probs, labels = ps.topk(topk)
    probs = np.array(probs)
    labels = np.array(labels)

    probs = probs[0]
    labels = labels[0]

    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }

    labels = [mapping [item] for item in labels]
    labels = np.array (labels)
   # return probs, flower_names
    return probs, labels






#===============================================================================
#running program
def main():
    ar = pass_args()
    device = availableDevice(ar.gpu)
    #model = load_checkpoint(ar.filepath)
    #imagepath = process_image(ar.image_path)
    #predicted = predict(imagepath, model)
    print(device)
if __name__ == '__main__' : main()
