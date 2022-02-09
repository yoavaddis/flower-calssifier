import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import OrderedDict
import json
from torch.autograd import Variable
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', default =  "flowers/test/15/image_06351.jpg",
                    help="The path to image we want to check")
parser.add_argument('--chkp', default = 'checkpoint.pth',
                    help="path to checkpoint ")
parser.add_argument('--top_k', default=1, type=int)
parser.add_argument('--jfile', default = './cat_to_name.json',
                    help="Caregory names file json")

def load_checkpoint(chkp):

    loaded_chkp = torch.load(chkp)

    if(checkpoint['arch'].lower() == 'vgg19' or checkpoint['arch'].lower() == 'densenet161'):
        model = getattr(torchvision.models, checkpoint['arch'])(pretrained = True)
    model.classifier = loaded_chkp['classifier']
    model.load_state_dict(loaded_chkp['state_dict'])
    model.class_to_idx = loaded_chkp['class_to_idx']
    for param in model.parameters():
        param.requires_grad = False
    print("Modle Loaded")
    return model


def image_proccess(img_dir):
    img = Image.open(img_dir)
    img = img.resize((256,256))
    img = img.crop((0,0,224,224))
    img = np.array(img)/255
    means = [0.485, 0.456, 0.406]
    standiv = [0.229, 0.224, 0.225]
    img = (img - means) / standiv
    img = img.transpose((2, 0, 1))

    return np.array(img)


def predict(categories, img_np, model,  topk):
    model.to('cpu')
    model.eval()

    with torch.no_grad():

        img = torch.from_numpy(np.array([img_np])).float()
        img.to('cpu')
        logps = model.forward(img)
        ps = torch.exp(logps)
        p, classes = ps.topk(topk)
        top_p = p.tolist()[0:1]
        top_classes = classes.tolist()[0:1]
        idx_to_class = {v:k for k, v in model.class_to_idx.items()}
        labels = []
        for c in top_classes:
            labels.append(categories[idx_to_class[c]])


        output = list(zip(top_p, labels))

        print("probs and kind of flower: {}".format(output))

def main():


    args = args_paser()
    data_dir = 'flowers'

    l_model = load_checkpoint(arg.chkp)
    with open(arg.jfile, 'r') as file:
        json_content = json.load(file)

    img_np = image_proccess(arg.img_dir)
    predict(args.jfile, img_np,l_model,args.topk)
if __name__ == '__main__': main()
