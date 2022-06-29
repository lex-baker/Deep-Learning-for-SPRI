import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torchvision.transforms as tt
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if __name__ == "__main__":

    data_dir = "./Desktop/AI Projects/8 Designing Our Own/data/"

    #stats is "maybe" needed
    #stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tfms = tt.Compose([tt.Resize((224, 224)), tt.RandomResizedCrop(224, scale=(.8, 1)), tt.ToTensor()]) 
    valid_tfms = tt.Compose([tt.Resize((224, 224)), tt.ToTensor()])

    #pytorch datasets
    train_ds = ImageFolder(data_dir+'/train', train_tfms) #transform=ToTensor()
    valid_ds = ImageFolder(data_dir+'/test', valid_tfms)

    batch_size = 64

    #pytorch dataloaders
    # PyTorch data loaders
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=2, pin_memory=True)

    ##########################
    #Mass copy paste below
    ##########################

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
            
        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl: 
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)

    device = get_default_device()
    print(device)

    valid_dl = DeviceDataLoader(valid_dl, device)

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    class ImageClassificationBase(nn.Module):
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
            return {'val_loss': loss.detach(), 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

    def conv_block(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    class ResNet9(ImageClassificationBase):
        def __init__(self, in_channels, num_classes):
            super().__init__()
            
            self.conv1 = conv_block(in_channels, 64)
            self.conv2 = conv_block(64, 128, pool=True)
            self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
            
            self.conv3 = conv_block(128, 256, pool=True)
            self.conv4 = conv_block(256, 512, pool=True)
            self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
            
            self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                            nn.Flatten(), 
                                            nn.Dropout(0.2),
                                            nn.Linear(25088, num_classes))
            
        def forward(self, xb):
            out = self.conv1(xb)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.classifier(out)
            return out

    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    model = to_device(ResNet9(3, 18), device)
    model.load_state_dict(torch.load("./Desktop/AI Projects/8 Designing Our Own/ResNet9/75-ResNet9.pth", map_location=torch.device("cpu")))

    history = [evaluate(model, valid_dl)]
    print(history)


    def plot_accuracies(history):
        accuracies = [x['val_acc'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs')
        plt.show()

    plot_accuracies(history)

    def plot_losses(history):
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.show()

    plot_losses(history)

    def plot_lrs(history):
        lrs = np.concatenate([x.get('lrs', []) for x in history])
        plt.plot(lrs)
        plt.xlabel('Batch no.')
        plt.ylabel('Learning rate')
        plt.title('Learning Rate vs. Batch no.')
        plt.show()

    plot_lrs(history)

    def predict_image(img, model):
        # Convert to a batch of 1
        xb = to_device(img.unsqueeze(0), device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds  = torch.max(yb, dim=1)
        # Retrieve the class label
        return train_ds.classes[preds[0].item()]

    from random import randint

    img, label = valid_ds[randint(0, (len(valid_ds) - 1))]
    plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
    plt.show()
    print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))

    img, label = valid_ds[randint(0, (len(valid_ds) - 1))]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    print('Label:', valid_ds.classes[label], ', Predicted:', predict_image(img, model))

    img, label = valid_ds[randint(0, (len(valid_ds) - 1))]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))