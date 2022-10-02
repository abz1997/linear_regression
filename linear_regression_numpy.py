import numpy as np

# for linear regression we use f = w*x (no bias in this example)

# data for x and y where f = 2*x
x = np.array([1,2,3,4])
y = np.array([2,4,6,8])

# we initialise the weight as w = 0 at the start and we want it to get to 2
w = 0

# model prediction
def forward(x):
    return w * x

# the loss function is MSE for linear regression
# returns an array of numbers without mean().
def loss(y,y_pred):
    return((y-y_pred)**2).mean()

# gradient
# np.dot is dot product
# MSE = 1/N * (w*x - y)**2
# dMSE/dw = 1/N 2x (w*x - y)
def gradient(x,y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

# if learning rate too large, we will overshoot and fail to converge
learning_rate = 0.01
iterations = 10

for epoch in range(iterations):
    # the prediction is forward(x)
    y_pred = forward(x)

    # loss function
    l = loss(y,y_pred)

    # gradient of loss with respect to w
    dw = gradient(x,y,y_pred)

    # after each iteration we update w (we want to get dw to 0 ie dMSE/dw = 0 where loss function is at minimum)
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w ={w:.4f}, loss = {l:.4f} , dw = {dw:.4f}, y_pred = {y_pred}')

predict_y = 6

print(f'Prediction after {iterations} iterations of training: f({predict_y}) = {forward(predict_y)}')