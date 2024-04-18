# from leetcode_w374 import sum

# print(sum(1, 4))

from engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1  
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db

import random
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from engine import Value
from nn import  MLP
# Neuron, Layer,
# from torch.nn import MLP

np.random.seed(1337)
random.seed(1337)

# make up a dataset

from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=100, noise=0.1)

# print(X, y)

# y = y*2 - 1 # make y be -1 or 1
# # visualize in 2D
# plt.figure(figsize=(5,5))
# plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')

# initialize a model 
model = MLP(2, [16, 16, 1]) # 2-layer neural network
# print(model)
print("number of parameters", len(model.parameters()))

# loss function
def loss(batch_size=None):
    
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    
    # forward the model to get scores
    scores = list(map(model, inputs))
    
    # svm "max-margin" loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

# total_loss, acc = loss()
# print(total_loss, acc)

total_loss, acc = loss()
print(total_loss, acc)

# optimization
# optimization
for k in range(100):
    
    # forward
    total_loss, acc = loss()
    
    # backward
    model.zero_grad()
    total_loss.backward()
    
    # update (sgd)
    # learning_rate = 1.0 - 90*k/100
    learning_rate = 0.01
    # print(learning_rate)
    # print(model.parameters())
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    # print(p.data)
    # print(learning_rate)
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")