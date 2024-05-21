import numpy as np
import pickle
from pathlib import Path
import time
import torch

class Data_Prep:
    """This class loads the data into sets of 100, sampling a certain number in the sequences of 100 """
    def __init__(self):
        self.train_4d
        self.train_4d

    def Reshape(self, split_size, sequence_length, train_data, test_data):
        w =train_data.shape[1]
        h=train_data.shape[2]
        assert train_data.shape[2]==test_data.shape[2]
        assert train_data.shape[1]==test_data.shape[1]

        self.train_4d=train_data.reshape(train_data.shape[0]//sequence_length,sequence_length,w,h)[:,::split_size,:,:][:,:-1,:,:]
        self.train2_4d=train_data.reshape(train_data.shape[0]//sequence_length,sequence_length,w,h)[:,::split_size,:,:][:,:-1,:,:]
        self.train_shift_4d=train_data.reshape(train_data.shape[0]//sequence_length,sequence_length,w,h)[:,::split_size,:,:][:,1:,:,:]

        self.test_4d = self.test_data.reshape(self.test_data.shape[0]//sequence_length,sequence_length,w,h)[:,::split_size,:,:][:,:-1,:,:]
        self.test_shift_4d= self.test_data.reshape(self.test_data.shape[0]//sequence_length,sequence_length,w,h)[:,::split_size,:,:][:,1:,:,:]
        assert (self.train_shift_4d[0][1].all() ==self.train_4d[0][2].all())
        assert (self.test_shift_4d[0][1].all() == self.test_4d[0][2].all())

        return self.train_4d, self.test_4d
    
    def DataLoading(self):
        trainloader = torch.utils.data.DataLoader(torch.tensor(self.train_4d,dtype=torch.float32),batch_size=16, shuffle = False)
        trainshiftloader = torch.utils.data.DataLoader(torch.tensor(self.train_4d,dtype=torch.float32),batch_size=16, shuffle = False)
        testloader = torch.utils.data.DataLoader(torch.tensor(self.test_4d,dtype=torch.float32),batch_size=16, shuffle = False)
        testshiftloader = torch.utils.data.DataLoader(torch.tensor(self.test_4d,dtype=torch.float32),batch_size=16, shuffle = False)

        return trainloader, trainshiftloader, testloader, testshiftloader
