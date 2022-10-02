# Pytorch pipeline
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimiser
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn

# for linear regression we use f = w*x (no bias in this example)

# data for x and y where f = 2*x
x = torch.tensor([1,2,3,4], dtype = torch.float32)
y = torch.tensor([2,4,6,8], dtype = torch.float32)

# we initialise the weight as w = 0 at the start and we want it to get to 2. we must write requires_grad = True to indicate whether a variable is trainable.
w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

# model prediction
def forward(x):
    return w * x

# the loss function is MSE for linear regression
# returns an array of numbers without mean().
'''def loss(y,y_pred):
    return((y-y_pred)**2).mean()'''

# we dont need to define loss manually anymore


# if learning rate too large, we will overshoot and fail to converge
learning_rate = 0.01
iterations = 20

# we can import MSE and stochastic gradient descent optimiser
loss = nn.MSELoss()
optimiser = torch.optim.SGD([w], lr = learning_rate)

for epoch in range(iterations):
    # the prediction is forward(x) (forwards pass)
    y_pred = forward(x)

    # loss function
    l = loss(y,y_pred)

    # gradient = backward pass (pytorch does this automatically for us)
    l.backward() #dl/dw

    # after each iteration we update w (we want to get dw to 0 ie dl/dw = 0 where loss function is at minimum)
    # no_grad() is a context manager and is used to prevent calculating gradients
    # all the results of the computations will have requires_grad=False, even if the inputs have requires_grad = True ie w.requires_grad = False
    # Notice that you won't be able to backpropagate the gradient to layers before the no_grad.
    '''
    with torch.no_grad(): 
        grad = w.grad
        w -= learning_rate * grad'''

    # update weights using optimiser 
    optimiser.step()

    # we want to return to 0 gradient after each iteration with pytorch. This method allows as to edit w in place ie this is same as w = w.grad.zero()
    '''w.grad.zero_()'''
    optimiser.zero_grad()

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w ={w:.4f}, loss = {l:.4f}, y_pred = {y_pred}')

predict_y = 6

print(f'Prediction after {iterations} iterations of training: f({predict_y}) = {forward(predict_y)}')