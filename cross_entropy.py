import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

# y must be one-hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]

Y = np.array([1, 0, 0])

# y_pred has probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

# Cross-entropy loss in PyTorch
loss = nn.CrossEntropyLoss()

# 3 samples
# each sample has 3 class probabilities
Y = torch.tensor([0])
# nsamples x nclasses = 1x3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())

print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)
# output: tensor([0])


## neural network with softmax
# multiclass problem

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU() #activation function
    self.l2 = nn.Linear(hidden_size, num_classes)


  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    # no softmax at the end
    return out

model = NeuralNet(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() # (applies Softmax)
print(model)

# optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
num_epochs = 100
for epoch in range(num_epochs):
  # forward pass and loss
  y_pred = model(X)
  loss = criterion(y_pred, Y)

  # backward pass
  loss.backward()

  # updates
  optimizer.step()

  # zero gradients
  optimizer.zero_grad()

# Prediction
# model.eval()
with torch.no_grad():
  y_pred = model(X)
  loss = criterion(y_pred, Y)
  _, predictions = torch.max(y_pred, 1)
  correct = (predictions == Y).sum()
  print(f'Loss: {loss.item()}, accuracy: {correct.item() * 100 / Y.size(0)}

# Binary classification problem
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU() #activation function
    self.l2 = nn.Linear(hidden_size, num_classes)


  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    # sigmoid at the end
    y_pred = torch.sigmoid(out)
    return y_pred

    ## return out

#model = NeuralNet(input_size=28*28, hidden_size=5, num_classes=2)

model =  
criterion = nn.CrossEntropyLoss() # (applies Softmax)
print(model)
