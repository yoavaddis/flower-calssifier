import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import json
from torch.autograd import Variable
import argparse
import os
import argparse



def args_paser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='the path where to save the checkpoint')
    parser.add_argument('--arch', default='vgg16', help='choose a pytorch model')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--gpu',type=bool, default=True )
    args = parser.parse_args()

    return args


def process_data(train_dir, test_dir, valid_dir):
#Define your transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_valid_transform = transforms.Compose([transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])



#  Load the datasets with ImageFolder
    image_datasets =  [datasets.ImageFolder(train_dir, transform=train_transform),
                       datasets.ImageFolder(valid_dir, transform=test_valid_transform),
                        datasets.ImageFolder(test_dir, transform=test_valid_transform)]
    return image_datasets[0], image_datasets[1], image_datasets[2]


def loaders(train_datasets, valid_datasets,test_datasets):
#Using the image datasets and the trainforms, define the dataloaders
    dataloaders =  [torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle = True),
                    torch.utils.data.DataLoader(valid_datasets, batch_size=64, ),
                    torch.utils.data.DataLoader(test_datasets, batch_size=64, )]

    return dataloaders[0], dataloaders[1], dataloaders[2]

def basic_model(arch):
# Load pretrained_network
    if arch == None or arch == 'vgg':
        load_model = models.vgg16(pretrained=True)
        print('Use vgg16')
    else:
        print('Please vgg16 or desnenet only, defaulting to vgg16')
        load_model = models.vgg16(pretrained=True)

    return load_model

def set_classifier(model, hidden_units):
    if hidden_units == None:
        hidden_units = 1024
    input = model.classifier[0].in_features

    model.classifier =  nn.Sequential(nn.Linear(input, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

    return model


def train_model(epochs, trainloaders, validloaders,gpu,model,optimizer,criterion):
    if type(epochs) == type(None):
        epochs = 10
        print("Epochs = 10")
    steps = 0
    if gpu == True:
        model.to('cuda')

    running_loss = 0
    print_every = 60
    for epoch in range(epochs):
        for inputs, labels in trainloaders:
            steps += 1
            if gpu==True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloaders:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Valid loss: {test_loss/len(validloaders):.3f}.."
                          f"Valid accuracy: {accuracy/len(validloaders):.3f}")
                    running_loss = 0
                model.train()
    return model

def valid_model(model, testloaders, gpu,criterion):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloaders:
            if gpu == True:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print('Accuracy: {:.3f} %'.format(100 * accuracy/len(testloaders)))




def save_checkpoint(model, train_datase,lr,epochs,criterion,optimizer,hidden, arch):
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'input': 25088,
                'output': 102,
                'hidden_units': hidden_units,
                'arch': arch,
                'learning_rate': lr,
                'classifier' : model.classifier,
                'epochs': epochs,
                'criterion': criterion,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}
    return torch.save(checkpoint, 'chkp.pth')



def main():

    args = args_paser()
    data_dir = 'flowers'

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_dataset,valid_dataset, test_dataset = process_data(train_dir, test_dir, valid_dir)
    trainloaders, validloaders ,testloaders = loaders( train_dataset,valid_dataset, test_dataset)
    model = basic_model(args.arch)
    for param in model.parameters():
        param.requires_grad = False
    model = set_classifier(model, args.hidden)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    trmodel = train_model(args.epochs,trainloaders, validloaders, args.gpu,model,optimizer,criterion)
    valid_model(trmodel, testloaders, args.gpu, criterion)
    save_checkpoint(trmodel, train_dataset, arg.lr,arg.epochs , criterion, optimizer,arg.hidden, arg.arch )
    print('Completed!')
if __name__ == '__main__': main()
