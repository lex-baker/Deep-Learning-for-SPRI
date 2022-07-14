import numpy as np
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
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder


from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier


def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

modelslist = [
    #'inception', 
    'squeezenet', 
    'efficientnet']

data_dir = "/kaggle/input/road-signs/data"

for m in modelslist:
    checkpoint = torch.load('/kaggle/input/models-unaugmented/' + m + '.pth', map_location=torch.device('cpu'))

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
    
    print(m)
    
    model_ft.load_state_dict(checkpoint['model_state_dict'])

    tfms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #pytorch datasets
    ds = ImageFolder(data_dir+'/val', tfms) #transform=ToTensor()

    dl = DataLoader(ds, batch_size=1, shuffle=False)

    images, labels = iter(dl).next()

    #print("True Image & True Label")
    #imshow(torchvision.utils.make_grid(images, normalize=True), [ds.classes[i] for i in labels])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft.load_state_dict(checkpoint['optimiser_state_dict'])

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    criterion = checkpoint['loss']

    model = model_ft.to(device)

    model = model.eval()

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer_ft,
        input_shape=(3, input_size, input_size),
        nb_classes=18,
    )

    atks = [
        FastGradientMethod(estimator=classifier, eps=.1),
        FastGradientMethod(estimator=classifier, eps=.3)
    ]

    """
    atks = [
        FGSM(model, eps=.1),
        FGSM(model, eps=.3),
        BIM(model, eps=.1, alpha=2/255, steps=100),
        BIM(model, eps=.3, alpha=2/255, steps=100),
        CW(model, c=1, lr=0.01, steps=100, kappa=0),
        PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True)
    ]

    atks = [
        FGSM(model, eps=8/255),
        BIM(model, eps=8/255, alpha=2/255, steps=100),
        RFGSM(model, eps=8/255, alpha=2/255, steps=100),
        CW(model, c=1, lr=0.01, steps=100, kappa=0),
        PGD(model, eps=8/255, alpha=2/225, steps=100, random_start=True),
        PGDL2(model, eps=1, alpha=0.2, steps=100),
        EOTPGD(model, eps=8/255, alpha=2/255, steps=100, eot_iter=2),
        FFGSM(model, eps=8/255, alpha=10/255),
        TPGD(model, eps=8/255, alpha=2/255, steps=100),
        MIFGSM(model, eps=8/255, alpha=2/255, steps=100, decay=0.1),
        VANILA(model),
        GN(model, std=0.1),
        APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
        APGD(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
        APGDT(model, eps=8/255, steps=100, eot_iter=1, n_restarts=1),
        FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=False),
        FAB(model, eps=8/255, steps=100, n_classes=10, n_restarts=1, targeted=True),
        Square(model, eps=8/255, n_queries=5000, n_restarts=1, loss='ce'),
        AutoAttack(model, eps=8/255, n_classes=10, version='standard'),
        OnePixel(model, pixels=5, inf_batch=50),
        DeepFool(model, steps=100),
        DIFGSM(model, eps=8/255, alpha=2/255, steps=100, diversity_prob=0.5, resize_rate=0.9)
    ]
    """

    #print("Adversarial Image & Predicted Label")

    for atk in atks :
        
        print("-"*70)
        print(atk)
        
        correct = 0
        total = 0
        
        for images, labels in dl:
            #outputs = torch.from_numpy(classifier.predict(x_test_adv))
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)
            total += 1
            labels.to(device)
            correct += (pre == labels).sum()
        #accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on benign test examples: {}%".format(100 * correct / total))
        
        total = 0
        correct = 0
        
        for images, labels in dl:
            # Step 6: Generate adversarial test examples
            x_test_adv = atk.generate(x=images.cpu().detach().numpy())
            #outputs = torch.from_numpy(classifier.predict(x_test_adv))
            outputs = model(torch.from_numpy(x_test_adv))
            _, pre = torch.max(outputs.data, 1)
            total += 1
            labels.to(device)
            correct += (pre == labels).sum()
        # Step 7: Evaluate the ART classifier on adversarial test examples

        
        #accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on adversarial test examples: {}%".format(100 * correct / total))