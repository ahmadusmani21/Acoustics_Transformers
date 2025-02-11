import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class WaveTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(WaveTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x

# Example usage
input_dim = 10  # Example: Wave parameters
model_dim = 64
num_heads = 4
num_layers = 3
output_dim = 10  # Example: Predicted wave properties

model = WaveTransformer(input_dim, model_dim, num_heads, num_layers, output_dim)

# Example input
x = torch.rand((32, 10, input_dim))  # Batch size of 32, sequence length of 10
output = model(x)
print(output.shape)  # Expected: (32, 10, output_dim)

# Training script
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            x, y = batch
            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Generating synthetic dataset
dataset_size = 1000
x_data = torch.rand((dataset_size, 10, input_dim))
y_data = torch.rand((dataset_size, 10, output_dim))
dataset = TensorDataset(x_data, y_data)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize training components
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer)

# Testing script
def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            predictions = model(x)
            print(f"Predictions: {predictions}, Ground Truth: {y}")

# Creating a test dataset
test_loader = DataLoader(dataset, batch_size=10, shuffle=False)
test_model(model, test_loader)