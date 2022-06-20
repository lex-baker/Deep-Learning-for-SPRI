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

    # Use a white background for matplotlib figures
    matplotlib.rcParams['figure.facecolor'] = '#ffffff'


    dataset = MNIST(root='Desktop/AI Projects/5 Neural Networks on the GPU/data/', download=True, transform=ToTensor())
    #print(len(dataset))

    image, label = dataset[0]
    #print('image.shape:', image.shape)
    #plt.imshow(image.permute(1, 2, 0), cmap='gray')
    #plt.show()
    #print('Label:', label)


    val_size = 10000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    #print("training and validating lengths", len(train_ds), len(val_ds))



    batch_size=128

    #pin_memory increases data tranfser speeds on CUDA-enabled GPUs
    #num_workers allows for parallelization of data loading (4 child process at int value of 4)
    #Removed the num_workers param because either python or vscode doesn't handle multithreading and async well
    #solution would be to use an async wait command on the for loop below
    #actual solution is to implement the overall if statement at the beginning of this program
    #It's odd that val_loader doubles the batch size, internet suggests that it is doubled internally, so this may be quadrupled
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

    """
    for images, _ in train_loader:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
        plt.show()
        break
    """

    for images, labels in train_loader:
        #print('images.shape:', images.shape)
        inputs = images.reshape(-1, 784)
        #print('inputs.shape:', inputs.shape)
        break

    print("="*16, "layer 1", "="*16)

    input_size = inputs.shape[-1]
    hidden_size = 32

    #returns a layer that takes all input and returns the size of the hidden middle layer
    layer1 = nn.Linear(input_size, hidden_size)

    print('inputs.shape:', inputs.shape)

    #this is the 
    layer1_outputs = layer1(inputs)
    print('layer1_outputs.shape:', layer1_outputs.shape)

    #this is for verification of above function only
    layer1_outputs_direct = inputs @ layer1.weight.t() + layer1.bias
    print(layer1_outputs_direct.shape)

    #This checks that the two methods are within a tolerance of 1e-3 of each other
    print("verification:",torch.allclose(layer1_outputs, layer1_outputs_direct, 1e-3))

    print("="*16, "layer 2", "="*16)

    #ReLU is the Rectified Linear Unit function
    #Literally just retuns 0 if given a negative, and the input if its above 0
    F.relu(torch.tensor([[1, -1, 0], [-0.1, .2, 3]]))

    #as said above, this function replaces all negative values with 0
    relu_outputs = F.relu(layer1_outputs)
    #.item() returns a single value tensor as a standard python number
    #these print statements show what ReLU function does
    #print('min(layer1_outputs):', torch.min(layer1_outputs).item())
    #print('min(relu_outputs):', torch.min(relu_outputs).item())

    #10, because there are only 10 options in the end
    output_size = 10
    layer2 = nn.Linear(hidden_size, output_size)

    #it's 128 because that's the batch size, and 10 because thats the number of outputs
    layer2_outputs = layer2(relu_outputs)
    print("output shape:",layer2_outputs.shape)

    #This calculates loss of final outputs when compared with labels-
    print("loss:",F.cross_entropy(layer2_outputs, labels))

    # Expanded version of layer2(F.relu(layer1(inputs)))
    #just another verification method
    outputs = (F.relu(inputs @ layer1.weight.t() + layer1.bias)) @ layer2.weight.t() + layer2.bias

    #compares outputs to a certain tolerance, (1e-3)
    #and here we verify
    print("verification:",torch.allclose(outputs, layer2_outputs, 1e-3))




    
    #Below is NOT NEURAL, it is ONLY LINEAR
    #It is an EXAMPLE of what the output would look like WITHOUT hidden layer
    print("="*16, "modeling linear WITHOUT ReLU", "="*16)

    # Same as layer2(layer1(inputs))
    outputs2 = (inputs @ layer1.weight.t() + layer1.bias) @ layer2.weight.t() + layer2.bias

    # Create a single layer to replace the two linear layers
    combined_layer = nn.Linear(input_size, output_size)

    combined_layer.weight.data = layer2.weight @ layer1.weight
    combined_layer.bias.data = layer1.bias @ layer2.weight.t() + layer2.bias

    # Same as combined_layer(inputs)
    outputs3 = inputs @ combined_layer.weight.t() + combined_layer.bias

    #Proving the two methods of purely linear relation are the same
    print("verification",torch.allclose(outputs2, outputs3, 1e-3))

    #loss of one iter of linear
    print("loss", F.cross_entropy(outputs2, labels))


    #time for the actual NEURAL MODEL
    #This class is the amalgamation of everything above 
    print("="*16, "making the model", "="*16)

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
            out = F.relu(out)
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
        
    #uhhh not part of the model class like I assumed
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


    #Real serious time now
    print("="*16, "preliminary testing", "="*16)
    input_size = 784
    hidden_size = 32 # you can change this
    num_classes = 10

    model = MnistModel(input_size, hidden_size=32, out_size=num_classes)

    for t in model.parameters():
        print(t.shape)
    
    #This is some basic outputs, with a mostly untrained model
    #...i think
    for images, labels in train_loader:
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        print('Loss:', loss.item())
        break

    print('outputs.shape : ', outputs.shape)
    print('Sample outputs :\n', outputs[:2].data)
    
    print("="*16, "gpu shenanigans", "="*16)
    #can i even use a gpu?
    print("cuda gpu?",torch.cuda.is_available())

    def get_default_device():
        #Pick GPU if available, else CPU
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    device = get_default_device()
    print(device)

    def to_device(data, device):
        #Move tensor(s) to chosen device
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    for images, labels in train_loader:
        print(images.shape)
        images = to_device(images, device)
        print("should match other",images.device)
        break


    #actual deep learning time
    print("="*16, "actual deep learning", "="*16)

    #new dataloader for the device
    #as shown by __init__ function, this is extending the functionality of DataLoader class
    class DeviceDataLoader():
        #Wrap a dataloader to move data to a device
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            #Yield a batch of data after moving it to device
            for b in self.dl:
                #Yield is like return, but can return multiple values in an iterable method
                yield to_device(b, self.device)

        def __len__(self):
            #Number of batches
            return len(self.dl)

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)

    #checking that our device class is working properly
    #protip, it is
    """
    for xb, yb in val_loader:
        print('xb.device:', xb.device)
        print('yb:', yb)
        break
    """

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

    #moving the data and parameters to the right device
    #not very useful rn because I'm on a laptop lol
    # Model (on GPU)
    model = MnistModel(input_size, hidden_size=hidden_size, out_size=num_classes)
    to_device(model, device)

    #fiiiinally testing (with initial, random weights)
    history = [evaluate(model, val_loader)]
    print("random weights:")
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
        xb = to_device(img.unsqueeze(0), device)
        yb = model(xb)
        _, preds  = torch.max(yb, dim=1)
        return preds[0].item()
    
    img, label = test_dataset[0]
    plt.imshow(img[0], cmap='gray')
    print('Label:', label, ', Predicted:', predict_image(img, model))
    plt.show()

    img, label = test_dataset[1839]
    plt.imshow(img[0], cmap='gray')
    print('Label:', label, ', Predicted:', predict_image(img, model))
    plt.show()

    img, label = test_dataset[193]
    plt.imshow(img[0], cmap='gray')
    print('Label:', label, ', Predicted:', predict_image(img, model))
    plt.show()

    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size=256), device)
    result = evaluate(model, test_loader)
    print(result)

    #actually finally save the model so i can stop training it over and over.
    torch.save(model.state_dict(), 'Desktop/AI Projects/5 Neural Networks on the GPU/mnist-feedforward.pth')