
#This is for fixing Omar's Resnet18 saved model to be compatible with my code
#Instead of changing everything I did, I'm simply going to modify his .pth file to add in the neccessary parts

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models



checkpoint = torch.load('Desktop/AI Projects/10 Adversarial Attacks/models/omar-resnet18.pt')

model = models.resnet18(pretrained = True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 18)

blankoptimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


#Finally actually saving the model
torch.save(
    {
        'model_state_dict': checkpoint,
        'optimiser_state_dict': blankoptimizer.state_dict(),
        'loss': nn.CrossEntropyLoss(),
        'input_size': 224,
        'model_structure': repr(model)
    },
    'Desktop/AI Projects/10 Adversarial Attacks/models/resnet.pth'
)