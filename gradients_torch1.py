# 1) design the model (input, output. forward pass)
# 2) construct losss and optimizer
# 3) training loop
#  -forward pass: compute prediction
#  -backward pass: gradients
#   -update weights
import torch
import torch.nn as nn

# Given data
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# Initialize weight
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Define forward function (model prediction)
def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training parameters
learning_rate = 0.01
n_iters = 100

# Define loss function
loss = nn.MSELoss()

# Define optimizer
optimizer = torch.optim.SGD([w], lr=learning_rate)

# Training loop
for epoch in range(n_iters):
    # Forward pass
    y_pred = forward(X)

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
        print(f'Epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

# Print prediction after training
print(f'Prediction after training: f(5) = {forward(5):.3f}')
