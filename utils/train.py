import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from livelossplot import PlotLosses
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

def split(arr, split_size):
    L = len(arr)
    num_splits = L // split_size
    remainder = L % split_size
    splits = np.split(arr[:L-remainder], num_splits)
    if remainder != 0:
        splits.append(arr[L-remainder:])
    return splits


def create_pairs(data, split_size):
    x = split(data, split_size)
    y = split(data, split_size)
    for i in range(len(x)):
        x[i] = x[i][:-1]
        y[i] = y[i][1:]
    x, y = np.array(x), np.array(y)
    return np.concatenate(x), np.concatenate(y)


def create_dataloader(path, batch_size, mode='train'):
    data = np.array(np.load(open(path, 'rb')))
    data_x, data_y = create_pairs(data, 100)
    tensor_x, tensor_y = torch.Tensor(data_x), torch.Tensor(data_y)
    dataset = TensorDataset(tensor_x, tensor_y)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode != 'val'))  # noqa
    return data_loader


def Reshape(split_size, sequence_length, train_data, test_data):
    w =train_data.shape[1]
    h=train_data.shape[2]
    assert train_data.shape[2]==test_data.shape[2]
    assert train_data.shape[1]==test_data.shape[1]

    train_4d=train_data.reshape(train_data.shape[0]//sequence_length,sequence_length,w,h)[:,::split_size,:,:][:,:-1,:,:]
    train2_4d=train_data.reshape(train_data.shape[0]//sequence_length,sequence_length,w,h)[:,::split_size,:,:][:,:-1,:,:]
    train_shift_4d=train_data.reshape(train_data.shape[0]//sequence_length,sequence_length,w,h)[:,::split_size,:,:][:,1:,:,:]

    test_4d = test_data.reshape(test_data.shape[0]//sequence_length,sequence_length,w,h)[:,::split_size,:,:][:,:-1,:,:]
    test_shift_4d= test_data.reshape(test_data.shape[0]//sequence_length,sequence_length,w,h)[:,::split_size,:,:][:,1:,:,:]
    assert (train_shift_4d[0][1].all() ==train_4d[0][2].all())
    assert (test_shift_4d[0][1].all() == test_4d[0][2].all())

    return train_4d,test_4d


def DataLoading(train_4d, test_4d):
    trainloader = torch.utils.data.DataLoader(torch.tensor(train_4d,dtype=torch.float32),batch_size=16, shuffle = False)
    trainshiftloader = torch.utils.data.DataLoader(torch.tensor(train_4d,dtype=torch.float32),batch_size=16, shuffle = False)
    testloader = torch.utils.data.DataLoader(torch.tensor(test_4d,dtype=torch.float32),batch_size=16, shuffle = False)
    testshiftloader = torch.utils.data.DataLoader(torch.tensor(test_4d,dtype=torch.float32),batch_size=16, shuffle = False)

    return trainloader, trainshiftloader, testloader, testshiftloader





def loss_function(x, x_hat, mu, logvar):
    # Flatten the input and output for binary cross-entropy loss calculation
    x = x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))
    x_hat = x_hat.view(-1, x_hat.size(1) * x_hat.size(2) * x_hat.size(3) * x_hat.size(4))
    
    # MSE loss
    # reproduction_loss = mseloss(x_hat, x)#, reduction='mean')
    reproduction_loss = nn.MSELoss(x_hat, x, reduction='mean')
    
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return reproduction_loss + KLD

def train_model(model, dataloader, testloader, num_epochs, lr = 0.0001, criterion = loss_function,  t = 99, device = 'cpu', scheduler = 0):
    mus= []
    sigmas = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    plot_losses = PlotLosses()

    for epoch in range(num_epochs):
        # Training loop for the model
        model.train()
        train_loss = 0
        for X, Y in zip(dataloader, testloader):
            X = X.to(device).view(-1,1,t,256,256)
            Y = Y.to(device).view(-1,1,t,256,256)
            optimizer.zero_grad()
            a2, mu, logvar =model(X)
            loss = criterion(Y, a2, mu, logvar)

            mus.append(mu)
            sigmas.append(logvar)

            loss.backward()
            train_loss += loss*X.size(0)
            optimizer.step()
            if scheduler !=0:
                scheduler.step()

        train_loss /= len(dataloader.dataset)
        train_losses.append(train_loss)
        
        # Validation loop for the model
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X, Y in zip(dataloader, testloader):
                X = X.to(device).view(-1,1,t,256,256)
                Y = Y.to(device).view(-1,1,t,256,256)
                a2, mu, logvar =model(X)
                loss = criterion(Y, a2, mu, logvar)
                valid_loss += loss*X.size(0)
            
            valid_loss /= len(testloader.dataset)
            val_losses.append(valid_loss)
        
        # Update the live loss plot
        plot_losses.update({
            'loss': train_loss,
            'val_loss': valid_loss
        })
        plot_losses.send()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}')

    # Save plot losses figure
    plot_filename = f'loss_plot_{num_epochs}.png'
    # Plot the losses using matplotlib
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss')
    plt.savefig(plot_filename)
    plt.close()

    # Save the model and execution time
    model_info = {'model': model.state_dict(), 
                  'plot_losses':plot_filename}
    
    model_filename = f'VAE_{num_epochs}ep.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model_info, f)
    print(f"Model saved to {model_filename}")