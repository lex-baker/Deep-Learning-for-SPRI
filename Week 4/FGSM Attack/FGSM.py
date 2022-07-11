import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
import os
import time
import copy

import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import matplotlib.pyplot as plt

import numpy as np

#import adversarial-robustness-toolbox as art

from art.attacks.evasion import FastGradientMethod


epsilons = [0, .05, .1, .15, .2, .25, .3]


#if __name__ == "__main__":

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = 'Desktop/AI Projects/10 Adversarial Attacks/'
folder_name = 'Squeezenet Stuff/'
model_name = '95-squeezenet.pth'

checkpoint = torch.load(os.path.join(data_dir, folder_name, model_name), map_location=torch.device('cpu'))
input_size = 224


valid_tfms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#pytorch datasets
valid_ds = datasets.ImageFolder(data_dir+'data/val', valid_tfms)

valid_dl = DataLoader(valid_ds, 1, num_workers=0, pin_memory=True)

model = torchvision.models.squeezenet1_1()

model.classifier[1] = nn.Conv2d(512, 18, kernel_size=(1,1), stride=(1,1))
model.num_classes = 18
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def eval_model(model, criterion=nn.CrossEntropyLoss()):
        #model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in valid_dl:
            #inputs = inputs.to(device)
            #labels = labels.to(device)

            # forward
            # Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)


            # statistics
            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data)
        
        loss = running_loss / len(valid_dl.dataset)
        print("correct", running_corrects.double().item())
        print("total", len(valid_dl.dataset))
        accuracy = running_corrects.double().item() / len(valid_dl.dataset)

        print("Loss =", loss)
        print("Accuracy =", accuracy)

eval_model(model)

print("beginning attack:")

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    #print(epsilon*sign_data_grad)
    perturbed_image = image + (epsilon*sign_data_grad)
    # Adding clipping to maintain [0,1] range (only when epsilon isn't 0)
    #if epsilon != 0: perturbed_image = torch.clamp(perturbed_image, 0, 1)
    #print('pre' * 100)
    #print(perturbed_image)
    #print('mid'*100)
    perturbed_image = torch.clamp(perturbed_image, -3, 3)
    #print('post'*100)
    #print(perturbed_image)
    # Return the perturbed image
    return perturbed_image

def test(model, device, test_loader, epsilon, criterion=nn.CrossEntropyLoss()):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        #print(init_pred)
        #print(target)
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        #I don't think this affects the model, but without it the program will not run
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data
        #print(data_grad)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader.dataset))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []


# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, valid_dl, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()