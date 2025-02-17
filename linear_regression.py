# 1) design model (input, output size, forward pass)
# 2) construct loss and optimizer
# 3) training loop
#   - forward pass: compute prediction and loss
#   - backward pass: gradients
#   - updates weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# preparing the data
X_numpy, Y_numpy = datasets.make_regression(n_samples = 100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape
# 1) model

input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

#loss

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass
    Y_predicted = model(X)
    loss = criterion(Y_predicted, Y)
    
    #backward pass
    loss.backward()
    
    #update
    optimizer.step()
    
    #zero gradients
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item()}')

#plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()