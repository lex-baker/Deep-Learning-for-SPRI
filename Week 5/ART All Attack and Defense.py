import numpy as np
#import cupy as cp
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import models
#import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder

from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

from art.defences.preprocessor import GaussianAugmentation, JpegCompression, FeatureSqueezing, SpatialSmoothingPyTorch, TotalVarMin

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

modelslist = [
    'inception', 
    'squeezenet',
    'efficientnet',
    'resnet'
    ]

data_dir = "/kaggle/input/road-signs/data"
#data_dir = "Desktop/AI Projects/10 Adversarial Attacks/data"

for m in modelslist:
    checkpoint = torch.load('/kaggle/input/models-unaugmented/' + m + '.pth', map_location=torch.device('cpu'))
    #checkpoint = torch.load('Desktop/AI Projects/10 Adversarial Attacks/models/' + m + '.pth', map_location=torch.device('cpu'))

    if m == "squeezenet":
        #Squeezenet
        model_ft = models.squeezenet1_1(pretrained=True)
        model_ft.classifier[1] = nn.Conv2d(512, 18, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = 18
        input_size = 224
    
    elif m == "efficientnet":
        #Efficientnet_b0
        model_ft = models.efficientnet_b7(pretrained=True)
        #model_ft.classifier[1].out_features = num_classes
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, 18)
        input_size = 224

    elif m == "inception":
        #Inception v3
        #Be careful, expects (299,299) sized images and has auxiliary output
        model_ft = models.inception_v3(pretrained=True)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, 18)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 18)
        input_size = 299
    elif m == "resnet":
        ###
        #This is Omar's Resnet18 model, that I used to continue testing defense strategies on
        ###
        model_ft = models.resnet18(pretrained = True)
        in_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_features, 18)
        input_size = 224
    
    print("*"*70)
    print(m)
    
    model_ft.load_state_dict(checkpoint['model_state_dict'])

    tfms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        #Removed normalize to utilize jpeg compression defense
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #pytorch datasets
    ds = ImageFolder(data_dir+'/val', tfms)

    dl = DataLoader(ds, batch_size=1, shuffle=False)
    
    
    images = []
    labels = []
    for imgs, lbls in dl:
        images.append(imgs.numpy())
        labels.append(lbls.numpy())
    images = np.array(images).squeeze()
    #print(np.shape(images))
    labels = np.array(labels).squeeze()
    #print(np.shape(images))
    labels = np.squeeze(np.eye(18)[labels.reshape(-1)])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft.load_state_dict(checkpoint['optimiser_state_dict'])

    # Setup the loss fxn
    #criterion = nn.CrossEntropyLoss()
    criterion = checkpoint['loss']

    model = model_ft.to(device)

    model = model.eval()
    
    defenses = [
        #GaussianAugmentation(),
        #LabelSmoothing(),
        #FeatureSqueezing(clip_values=(0, 1)),
        TotalVarMin(verbose=True),
        SpatialSmoothingPyTorch(),
        #GaussianAugmentation(clip_values=(0,1))
        #FeatureSqueezing(clip_values=(0, 1)) # or (0, 255) idk
        #JpegCompression(clip_values=(0, 255), channels_first=True)
    ]

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer_ft,
        input_shape=(3, input_size, input_size),
        nb_classes=18,
        #device_type="gpu",
        #preprocessing_defences=TotalVarMin()
        #post_processing_defenses=
    )

    atks = [
        FastGradientMethod(estimator=classifier, eps=0.4, batch_size=16),
        BasicIterativeMethod(estimator=classifier, eps=0.4, batch_size=16)
        #these are the default values for PGD in ART
        ProjectedGradientDescentPyTorch(estimator=classifier,eps=0.3,eps_step=0.1, batch_size=16, verbose=True)
    ]
    
    #######################No attacks no defenses
    
    print('-'*70)
    print('No Attack, No Defenses')
    predictions = classifier.predict(x=images)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / len(labels)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    
    torch.cuda.empty_cache()
    
    ########################Defenses with no attacks
    
    print('-'*70)
    print('No Attack, All Defenses')
    
    pre_images = images
    for defs in defenses:
        pre_images = defs(pre_images)[0]
    
    predictions = classifier.predict(x=pre_images)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / len(labels)
    print("Accuracy on benign, defended test examples: {}%".format(accuracy * 100))

    torch.cuda.empty_cache()
    
    for atk in atks :
        ########################Attack with no defenses
        print('-'*70)
        print(type(atk).__name__ + ", No Defenses")
        #print('PGD Attack, No Defenses')

        adv_images = atk.generate(x=images)

        predictions = classifier.predict(x=adv_images)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / len(labels)
        print("Accuracy on " + type(atk).__name__ + "-attacked, undefended test examples: {}%".format(accuracy * 100))

        torch.cuda.empty_cache()
        ########################Attack with all defenses

        print("-"*70)
        print(type(atk).__name__ + ", All Defenses")
        #print('PGD Attack, All Defenses')
        
        #Remove this line if "attack with no defenses" is active
        #adv_images = atk.generate(x=images)
        
        for defs in defenses:
            adv_images = defs(adv_images)[0]
        
        predictions = classifier.predict(adv_images)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) / len(labels)
        print("Accuracy on " + type(atk).__name__ + "-attacked and defended test examples: {}%".format(accuracy * 100))

        torch.cuda.empty_cache()
    print("*"*70)