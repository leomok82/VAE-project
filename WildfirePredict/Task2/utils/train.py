import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn

def create_split(data, seq_length, split_size):
    dim = data.shape[1]
    nb_seq = data.shape[0] // seq_length
    data_4d = data.reshape(nb_seq, seq_length, dim, dim)[:,::split_size,:,:]
    return data_4d

def create_shifted_sequences(data, seq_length, split_size):
    split_data = create_split(data, seq_length, split_size)

    input_seq = split_data[:, :-1, :, :]
    target_seq = split_data[:, 1:, :, :]

    return input_seq, target_seq

class WildfiresObjective2(Dataset):
    def __init__(self, data, seq_length, split_size):
        self.seq_length = seq_length
        self.split_size = split_size
        self.input_seq, self.target_seq = create_shifted_sequences(data, seq_length, split_size)

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        return self.input_seq[idx], self.target_seq[idx]

def create_dataloaders(train_data, test_data, seq_length, split_size, batch_size):

    # Create datasets
    train_dataset = WildfiresObjective2(train_data, seq_length, split_size)
    test_set = WildfiresObjective2(test_data, seq_length, split_size)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def Reshape(split_size, sequence_length, train_data, test_data):
    w = train_data.shape[1]
    h = train_data.shape[2]
    
    train_4d = train_data.reshape(train_data.shape[0]//sequence_length, sequence_length, w, h)[:,::split_size,:,:][:,:-1,:,:]
    train_shift_4d = train_data.reshape(train_data.shape[0]//sequence_length, sequence_length, w, h)[:,::split_size,:,:][:,1:,:,:]

    test_4d = test_data.reshape(test_data.shape[0]//sequence_length, sequence_length, w, h)[:,::split_size,:,:][:,:-1,:,:]
    test_shift_4d = test_data.reshape(test_data.shape[0]//sequence_length, sequence_length, w, h)[:,::split_size,:,:][:,1:,:,:]

    return train_4d, test_4d

def DataLoading(train_4d, test_4d):
    trainloader = torch.utils.data.DataLoader(torch.tensor(train_4d,dtype=torch.float32), batch_size=16, shuffle=False)
    testloader = torch.utils.data.DataLoader(torch.tensor(test_4d,dtype=torch.float32), batch_size=16, shuffle=False)

    return trainloader, testloader

def loss_function(x, x_hat, mu, logvar):
    x = x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))
    x_hat = x_hat.view(-1, x_hat.size(1) * x_hat.size(2) * x_hat.size(3) * x_hat.size(4))
    
    mse_loss = nn.MSELoss(reduction='mean')
    reproduction_loss = mse_loss(x_hat, x)
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return reproduction_loss + KLD

def train(model, data_loader, data_shifted_loader, lr=0.0001, criterion=loss_function, t=19, device='cpu', scheduler=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    train_loss = 0
    for X, Y in zip(data_loader, data_shifted_loader):
        X = X.to(device).view(-1,1,t,256,256)
        Y = Y.to(device).view(-1,1,t,256,256)
        optimizer.zero_grad()
        a2, mu, logvar = model(X)
        loss = criterion(Y, a2, mu, logvar)
        loss.backward()
        train_loss += loss * X.size(0)
        optimizer.step()
        if scheduler != 0:
            scheduler.step()
    train_loss /= len(data_loader.dataset)
    return train_loss

def validate(model, data_loader, data_shifted_loader, criterion=loss_function, t=19, device='cpu'):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for X, Y in zip(data_loader, data_shifted_loader):
            X = X.to(device).view(-1,1,t,256,256)
            Y = Y.to(device).view(-1,1,t,256,256)
            a2, mu, logvar = model(X)
            loss = criterion(Y, a2, mu, logvar)
            valid_loss += loss * X.size(0)
        
        valid_loss /= len(data_loader.dataset)
    return valid_loss