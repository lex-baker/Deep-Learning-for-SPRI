from paddle import Paddle

import random
import numpy as np
#from keras import Sequential
from collections import deque
#from keras.layers import Dense
import matplotlib.pyplot as plt
#from keras.optimizers import Adam

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as tt

env = Paddle()
np.random.seed(0)


action_space = 3
state_space = 5
epsilon = 1
gamma = .95
batch_size = 64
epsilon_min = .01
epsilon_decay = .995
learning_rate = 0.001
memory = deque(maxlen=100000)


class DQN(nn.Module):

    """ Implementation of deep q learning algorithm """

    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    """
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
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
    """



def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

##########################################################################
#my (hopefully) working implementation of Keras' predict function
def predict(x, model):
    #xb = x.unsqueeze(0)
    #yb = self.model(xb)
    yb = model(x)
    _, preds  = torch.max(yb, dim=1)
    return preds

#my implementation of keras fit function
def fit(model, inputs, targets, epochs):
    #Train the model using gradient descent
    history = []
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epoch in range(epochs):
        # Training Phase 
        #for input in inputs:
        out = model(inputs)                  # Generate predictions
        loss = F.cross_entropy(out, targets) # Calculate loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    """
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
    """

def act(state, model):
    if np.random.rand() <= epsilon:
        return random.randrange(action_space)
    act_values = predict(state, model)
    return np.argmax(act_values[0])

def replay(model):

    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)
    states = np.array([i[0] for i in minibatch])
    actions = np.array([i[1] for i in minibatch])
    rewards = np.array([i[2] for i in minibatch])
    next_states = np.array([i[3] for i in minibatch])
    dones = np.array([i[4] for i in minibatch])

    states = np.squeeze(states)
    next_states = np.squeeze(next_states)

    targets = rewards + gamma*(np.amax(predict(next_states, model), axis=1))*(1-dones)
    targets_full = predict(states, model)

    ind = np.array([i for i in range(batch_size)])
    targets_full[[ind], [actions]] = targets

    fit(model, states, targets_full, epochs=1)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay




def train_dqn(episode):

    loss = []

    max_steps = 1000

    agent = DQN(state_space, action_space)
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, state_space))
        score = 0
        for i in range(max_steps):
            action = act(state, agent)
            reward, next_state, done = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, state_space))
            remember(state, action, reward, next_state, done)
            state = next_state
            replay(agent)
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)
    return loss


if __name__ == '__main__':

    ep = 100
    loss = train_dqn(ep)
    plt.plot([i for i in range(ep)], loss)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()