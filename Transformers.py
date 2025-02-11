import torch
import torch.nn as nn
import torch.optim as optim

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
