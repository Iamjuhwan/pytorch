# automation of pytorch
# 1) design the model (input, output. forward pass)
# 2) construct losss and optimizer
# 3) training loop
#  -forward pass: compute prediction
#  -backward pass: gradients
#   -update weights

import torch
import torch.nn as nn

# Given data
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_sample, n_features = X.shape
print( n_sample, n_features)

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size) #this function covers dor the forward and weight function - it is automation

class LinearRegression(nn.Module):

  def __init__(self, input_dim, output_dim):
    super(LinearRegression, self).__init__() 
    # define the layers
    self.lin = nn.Linear(input_dim, output_dim)
  
  def forward(self, x):
    return self.lin(x)

model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training parameters
learning_rate = 0.01
n_iters = 100

# Define loss function
loss = nn.MSELoss()

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(n_iters):
    # Forward pass
    y_pred = model(X)

    # Calculate loss
    l = loss(Y, y_pred)

    # Backward pass
    l.backward()  # Compute gradients dl/dw

    # Update weights
    optimizer.step()

    # Zero gradients
    optimizer.zero_grad()

    # Print progress
    if epoch % 10 == 0:
      [w, b] = model.parametrs()
        print(f'Epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

# Print prediction after training
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

