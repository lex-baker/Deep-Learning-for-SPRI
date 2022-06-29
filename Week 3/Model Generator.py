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

os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(42)

data_dir = "./Desktop/AI Projects/10 Best Model Development/"
model_folder = "Trained Models/"

class ClassicalTransferLearningModel(nn.Module):
    def __init__(self, num_of_features, n_classes):
        super().__init__()
        # print("num of features",num_of_features)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(num_of_features,32)
        self.relu1 = nn.ReLU6()
        self.fc2 = nn.Linear(32,n_classes)
        
    def forward(self,x):
        # x = self.tanh(x)
        
        x = self.fc1(x)
        #x = self.tanh(x)
        x = self.relu1(x)
        x = self.fc2(x)
        #print(x)
        #print("softmax")
        #print(F.softmax(x,dim=1))
        x = F.softmax(x,dim=1)
        
        return x

num_ftrs = 0

class classicalModel:
    model_classical = None
       
    def __init__(self, model_name = "resnet18",n_classes= 16):
        print(model_name)
        #global model_classical 
        if model_name == "resnet18":
            #geting the pre-trained model
            self.model_classical = torchvision.models.resnet18(pretrained=True)
            #freezing the initial layers of pre-trained models
            for param in self.model_classical.parameters():
                param.requires_grad = False
                
            #taking  the number of features from last layer of pre-trained models before the custom layer
            num_ftrs = self.model_classical.fc.in_features
            self.model_classical.fc = ClassicalTransferLearningModel(num_ftrs,n_classes)
    
        elif model_name == "alexnet":
            self.model_classical = torchvision.models.alexnet(pretrained=True)
            
            for param in self.model_classical.parameters():
                param.requires_grad = False
                
            num_ftrs =  self.model_classical.classifier[1].in_features
            self.model_classical.classifier = ClassicalTransferLearningModel(num_ftrs,n_classes)
    
            
        elif model_name == "vgg16":
            self.model_classical = torchvision.models.vgg16(pretrained=True)
            
            for param in self.model_classical.parameters():
                param.requires_grad = False
                
            num_ftrs = self.model_classical.classifier[0].in_features
            self.model_classical.classifier = ClassicalTransferLearningModel(num_ftrs,n_classes)
    
        elif model_name == "inception":
            self.model_classical = torchvision.models.inception_v3(pretrained=True)
            for param in self.model_classical.parameters():
                param.requires_grad = False
            
            num_ftrs = self.model_classical.fc.in_features
            self.model_classical.fc = ClassicalTransferLearningModel(num_ftrs,n_classes)
        
        elif model_name == "efficientnet_b0":
            self.model_classical = torchvision.models.efficientnet_b0(pretrained=True)
            for param in self.model_classical.parameters():
                param.requires_grad = False
            
            num_ftrs = self.model_classical.classifier[1].in_features
            self.model_classical.classifier = ClassicalTransferLearningModel(num_ftrs,n_classes)


        print(self.model_classical)
        
    def get_model(self):
        return self.model_classical

def train_model(model, criterion, optimizer, scheduler, num_epochs, name):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = 10000.0  # Large arbitrary number
        best_acc_train = 0.0
        best_loss_train = 10000.0  # Large arbitrary number
        print("Training started:")
#         dataloaders = self.dataset_dataloaders()
#         dataset_sizes = self.dataset_sizes()
        
        
        for epoch in range(num_epochs):
    
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    # Set model to training mode
                    model.train()
                else:
                    # Set model to evaluate mode
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
#                n_batches = self.dataset_sizes()[phase] // batch_size
#                it = 0
#                
                for inputs, labels in dataloaders[phase]:
#                    since_batch = time.time()
                    batch_size_ = len(inputs)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
    
                   
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = None
                        if model_name == "inception_v3" and phase== "train":
                            outputs = model(inputs)
                            outputs = outputs[0]
                        else:
                            outputs = model(inputs)
        #                     print(outputs)
        #                     print(labels)
        #                     break
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
    
                    # Print iteration results
                    running_loss += loss.item() * batch_size_
                    batch_corrects = torch.sum(preds == labels.data).item()
                    running_corrects += batch_corrects
#                     print(
#                        "Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}\n".format(
#                            phase,
#                            epoch + 1,
#                            num_epochs,
#                            it + 1,
#                            n_batches + 1,
#                            time.time() - since_batch,
#                        ),
#                        end="\r",
#                        flush=True,
#                    )
#                     it += 1
    
                # Print epoch results
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                print(
                    "Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}        ".format(
                        "train" if phase == "train" else "val \n ",
                        epoch + 1,
                        num_epochs,
                        epoch_loss,
                        epoch_acc,
                    )
                )
    
                # Check if this is the best model wrt previous epochs
                if phase == "val" and epoch_acc > best_acc:
                    print("saving model...")
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': criterion,
                            'classifier': ClassicalTransferLearningModel(num_ftrs, n_classes)
                        }, 
                        os.path.join(data_dir, model_folder, int(epoch_acc) + name + '.pt')
                    )
                    #torch.save(copy.deepcopy(model.state_dict()), data_dir + model_folder + "/" + int(epoch_acc) + name + '.pt')
                    #torch.save(copy.deepcopy(model.state_dict()), data_dir + model_folder +"/"+name+'.pth')
                    #torch.save(copy.deepcopy(model.state_dict()), data_dir + model_folder +"/"+name+'.pt')
                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                if phase == "train" and epoch_acc > best_acc_train:
                    best_acc_train = epoch_acc
                if phase == "train" and epoch_loss < best_loss_train:
                    best_loss_train = epoch_loss
    
                # Update learning rate
                if phase == "train":
                    scheduler.step()
    
        # Print final results
        model.load_state_dict(best_model_wts)
        time_elapsed = time.time() - since
        print(
            "\n\nTraining completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
        )
        print("\n\nBest test loss: {:.4f} | Best test accuracy: {:.4f}".format(best_loss, best_acc))
        return model
    

#data_dir = "my_data"     # folder location for data
model_name = "resnet18"  # model name 
batch_size = 8

opt_name= 'Adam'
lr = 0.0004
gamma_lr_scheduler = 0.1
step_size = 10
num_epochs = 25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dataset_transform(model_name):
        
        data_transforms ={}
        if model_name == "inception_v3":
            data_transforms = {
                "train": transforms.Compose(
                    [
                        # transforms.RandomResizedCrop(224),     # uncomment for data augmentation
                        # transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        # Normalize input channels using mean values and standard deviations of ImageNet.
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                ),
                "val": transforms.Compose(
                    [
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                ),
                }
        else:
            data_transforms = {
                "train": transforms.Compose(
                    [
                        # transforms.RandomResizedCrop(224),     # uncomment for data augmentation
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        # Normalize input channels using mean values and standard deviations of ImageNet.
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                ),
                "val": transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                ),
                }
        
        return data_transforms

image_datasets = {
        x if x == "train" else "val": datasets.ImageFolder(
            os.path.join(data_dir, "data", x), dataset_transform(model_name)[x]
        )
        for x in ["train", "val"]
    }


dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

class_names = image_datasets["train"].classes

dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
            for x in ["train", "val"]
        }

n_classes = len(class_names)


model_name = 'vgg16'

model = classicalModel(model_name,n_classes).get_model()

def get_optima_optimizer(model, optimizer_name: str ="Adam", lr: int = 0.0001):
    optimizer = getattr(
        torch.optim, optimizer_name
    )(filter(lambda p:p.requires_grad, model.parameters()), lr)

    return optimizer

def get_optima_Scheduler(optimizer, step_size: int = 10, gamma_lr_schedular: float=0.1):
    scheduler = lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma_lr_schedular
    )
    return scheduler

def get_loss_func(loss_func_name = 'cross_entropy'):
    loss_fn = None
    if loss_func_name == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
        
    print("Loss Function is: ", loss_fn)
    return loss_fn, loss_func_name


criterion, loss_func_name = get_loss_func()

optimizer = get_optima_optimizer(model, opt_name, lr)

scheduler = get_optima_Scheduler(optimizer, step_size, gamma_lr_scheduler)

best_model = train_model(model, criterion, optimizer, scheduler, 5, model_name+"name")