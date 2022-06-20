import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
#%matplotlib inline

if __name__ == '__main__':

    matplotlib.rcParams['figure.facecolor'] = '#ffffff'

    dataset = MNIST(root='Desktop/AI Projects/5 Neural Networks on the GPU/data/', download=True, transform=ToTensor())

    val_size = 10000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size=128

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)

    for images, labels in train_loader:
        inputs = images.reshape(-1, 784)
        break

    class MnistModel(nn.Module):
        #Feedfoward neural network with 1 hidden layer
        def __init__(self, in_size, hidden_size, out_size):
            super().__init__()
            # hidden layer
            self.linear1 = nn.Linear(in_size, hidden_size)
            # output layer
            self.linear2 = nn.Linear(hidden_size, out_size)
            
        def forward(self, xb):
            # Flatten the image tensors
            xb = xb.view(xb.size(0), -1)
            # Get intermediate outputs using hidden layer
            out = self.linear1(xb)
            # Apply activation function
            out = F.leaky_relu(out)
            # Get predictions using output layer
            out = self.linear2(out)
            return out
        
        def training_step(self, batch):
            images, labels = batch 
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss
        
        def validation_step(self, batch):
            images, labels = batch 
            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = accuracy(out, labels)           # Calculate accuracy
            return {'val_loss': loss, 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    input_size = 784
    hidden_size = 64
    num_classes = 10

    #Functions for training and testing model
    def evaluate(model, val_loader):
        #Evaluate the model's performance on the validation set
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        #Train the model using gradient descent
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            # Training Phase 
            for batch in train_loader:
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            result = evaluate(model, val_loader)
            model.epoch_end(epoch, result)
            history.append(result)
        return history

    # Model
    model = MnistModel(input_size, hidden_size, out_size=num_classes)

    #Testing
    history = [evaluate(model, val_loader)]
    print("With random weights:")
    print(history)

    #for learning
    #Train the model for 5 epochs at a learning rate of .5
    print("5 epochs, .5 LR:")
    history += fit(5, 0.5, model, train_loader, val_loader)
    
    #continue training the model for 5 more epochs at a learning rate of .1
    print("5 more epochs, .1 LR:")
    history += fit(5, 0.1, model, train_loader, val_loader)
    
    #graphing how everything went
    #for loss
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');
    plt.show()

    #for accuracy
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()


    #final testing with finalized model and overall accuracy established
    # Define test dataset
    test_dataset = MNIST(root='Desktop/AI Projects/5 Neural Networks on the GPU/data/', train=False, transform=ToTensor())

    def predict_image(img, model):
        xb = img.unsqueeze(0)
        yb = model(xb)
        _, preds  = torch.max(yb, dim=1)
        return preds[0].item()
    
    img, label = test_dataset[0]
    plt.imshow(img[0], cmap='gray')
    print('Label:', label, ', Predicted:', predict_image(img, model))
    #plt.show()

    img, label = test_dataset[1839]
    plt.imshow(img[0], cmap='gray')
    print('Label:', label, ', Predicted:', predict_image(img, model))
    #plt.show()

    img, label = test_dataset[193]
    plt.imshow(img[0], cmap='gray')
    print('Label:', label, ', Predicted:', predict_image(img, model))
    #plt.show()

    test_loader = DataLoader(test_dataset, batch_size=256)
    result = evaluate(model, test_loader)
    print(result)