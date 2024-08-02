import torch

# Given data
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# Initialize weight
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Define forward function (model prediction)
def forward(x):
    return w * x

# Define loss function (MSE)
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training parameters
learning_rate = 0.01
n_iters = 100

# Training loop
for epoch in range(n_iters):
    # Forward pass
    y_pred = forward(X)

    # Calculate loss
    l = loss(Y, y_pred)

    # Backward pass
    l.backward()  # Compute gradients dl/dw

    # Update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # Zero gradients
    w.grad.zero_()

    # Print progress
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

# Print prediction after training
print(f'Prediction after training: f(5) = {forward(5):.3f}')
