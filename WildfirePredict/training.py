import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from livelossplot import PlotLosses

from WildfirePredict.model import ConvLSTMModel

class WildfireTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, criterion, optimizer, num_epochs=20):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

        self.model.to(self.device)
        self.liveplot = PlotLosses()

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0

            tqdm_train_loader = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)

            for x_batch, y_batch in tqdm_train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * x_batch.size(0)
                tqdm_train_loader.set_postfix({'train_loss': train_loss / ((tqdm_train_loader.n + 1) * x_batch.size(0))})

            train_loss /= len(self.train_loader.dataset)

            val_loss = self.validate()

            self.liveplot.update({'log loss': train_loss, 'val_log loss': val_loss})
            self.liveplot.draw()

        print("Training complete")

    def validate(self):
        self.model.eval()
        val_loss = 0.0

        tqdm_val_loader = tqdm(self.val_loader, desc=f"Validation", leave=False)

        with torch.no_grad():
            for x_batch, y_batch in tqdm_val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)

                val_loss += loss.item() * x_batch.size(0)
                tqdm_val_loader.set_postfix({'val_loss': val_loss / ((tqdm_val_loader.n + 1) * x_batch.size(0))})

        val_loss /= len(self.val_loader.dataset)
        return val_loss

    def test(self):
        self.model.eval()
        test_loss = 0.0

        tqdm_test_loader = tqdm(self.test_loader, desc=f"Testing", leave=False)

        with torch.no_grad():
            for x_batch, y_batch in tqdm_test_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)

                test_loss += loss.item() * x_batch.size(0)
                tqdm_test_loader.set_postfix({'test_loss': test_loss / ((tqdm_test_loader.n + 1) * x_batch.size(0))})

        test_loss /= len(self.test_loader.dataset)
        print(f"Test Loss: {test_loss:.4f}")
        return test_loss


if __name__ == '__main__': 
    train_dataset = None
    val_dataset = None
    test_dataset = None

    # Assuming you have the following parameters
    input_dim = 1
    hidden_dim = [64, 64]
    kernel_size = (3, 3)
    num_layers = 2
    output_dim = 1

    # Create the model
    model = ConvLSTMModel(input_dim, hidden_dim, kernel_size, num_layers, output_dim)

    # Assuming you have train_loader, val_loader, and test_loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set device, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create the trainer
    trainer = WildfireTrainer(model, train_loader, val_loader, test_loader, device, criterion, optimizer, num_epochs=20)

    # Train the model
    trainer.train()

    # Test the model
    trainer.test()

