import unittest
import numpy as np
import torch
from utils.train import create_split, create_shifted_sequences, WildfiresObjective2, create_dataloaders, Reshape, DataLoading, loss_function, train, validate

class TestSplitData(unittest.TestCase):

    def setUp(self):
        self.data = np.random.rand(100, 256, 256)
        self.seq_length = 10
        self.split_size = 2
        self.batch_size = 4
        self.train_data = np.random.rand(1000, 256, 256)
        self.test_data = np.random.rand(200, 256, 256)
        self.device = 'cpu'
        
    def test_create_split(self):
        split_data = create_split(self.data, self.seq_length, self.split_size)
        self.assertEqual(split_data.shape[1], self.seq_length // self.split_size)

    def test_create_shifted_sequences(self):
        input_seq, target_seq = create_shifted_sequences(self.data, self.seq_length, self.split_size)
        self.assertEqual(input_seq.shape[1], self.seq_length // self.split_size - 1)
        self.assertEqual(target_seq.shape[1], self.seq_length // self.split_size - 1)

    def test_wildfires_objective2(self):
        dataset = WildfiresObjective2(self.data, self.seq_length, self.split_size)
        self.assertEqual(len(dataset), len(dataset.input_seq))
        self.assertEqual(len(dataset), len(dataset.target_seq))

    def test_create_dataloaders(self):
        train_loader, test_loader = create_dataloaders(self.train_data, self.test_data, self.seq_length, self.split_size, self.batch_size)
        self.assertEqual(len(train_loader.dataset), len(self.train_data) // (self.seq_length * self.split_size))
        self.assertEqual(len(test_loader.dataset), len(self.test_data) // (self.seq_length * self.split_size))

    def test_reshape(self):
        train_4d, test_4d = Reshape(self.split_size, self.seq_length, self.train_data, self.test_data)
        self.assertEqual(train_4d.shape[1], self.seq_length // self.split_size - 1)
        self.assertEqual(test_4d.shape[1], self.seq_length // self.split_size - 1)

    def test_data_loading(self):
        train_4d, test_4d = Reshape(self.split_size, self.seq_length, self.train_data, self.test_data)
        trainloader, testloader = DataLoading(train_4d, test_4d)
        self.assertEqual(len(trainloader.dataset), train_4d.shape[0])
        self.assertEqual(len(testloader.dataset), test_4d.shape[0])

    def test_loss_function(self):
        x = torch.rand(4, 1, 19, 256, 256)
        x_hat = torch.rand(4, 1, 19, 256, 256)
        mu = torch.rand(4, 1, 19, 256, 256)
        logvar = torch.rand(4, 1, 19, 256, 256)
        loss = loss_function(x, x_hat, mu, logvar)
        self.assertTrue(isinstance(loss.item(), float))

    def test_train(self):
        model = nn.Module()  # Use a dummy model for testing
        train_loader, test_loader = create_dataloaders(self.train_data, self.test_data, self.seq_length, self.split_size, self.batch_size)
        train_loss = train(model, train_loader, train_loader, device=self.device)
        self.assertTrue(isinstance(train_loss.item(), float))

    def test_validate(self):
        model = nn.Module()  # Use a dummy model for testing
        train_loader, test_loader = create_dataloaders(self.train_data, self.test_data, self.seq_length, self.split_size, self.batch_size)
        valid_loss = validate(model, test_loader, test_loader, device=self.device)
        self.assertTrue(isinstance(valid_loss.item(), float))

if __name__ == '__main__':
    unittest.main()