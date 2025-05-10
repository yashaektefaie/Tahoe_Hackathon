import anndata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Generate random sample data - 2000 samples, 1280 dimensions each
num_samples = 2000
dimension = 1280

# Generate random input vectors X
X = np.random.randn(num_samples, dimension)

# Generate target vectors Y (could be random or a function of X)
# Option 1: Completely random Y
# Y = np.random.randn(num_samples, dimension)

# Option 2: Y as a noisy function of X (more realistic for regression task)
W = np.random.randn(dimension, dimension) * 0.1  # Random weight matrix
b = np.random.randn(dimension) * 0.1  # Random bias
noise = np.random.randn(num_samples, dimension) * 0.05  # Random noise
Y = X @ W + b + noise  # Y = XW + b + noise

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Option 1: Simple Linear Regression model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1280, 1280)
    
    def forward(self, x):
        return self.linear(x)

# Option 2: Neural Network with hidden layers
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_dim=512):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1280, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1280)
        )
    
    def forward(self, x):
        return self.network(x)

# Choose which model to use
# model = LinearModel()
model = NeuralNetwork()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Print progress
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

# After training, to use the model for prediction:
def predict(input_vector):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        return model(input_tensor).numpy()
