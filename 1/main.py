import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as p

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

def mse(x, y, theta):
    return (1/x.size)*np.sum(np.square((x * theta[1] + theta[0]) - y))


# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# calculate closed-form solution

m = x_train.size

X = np.ones((m, 1))
X = np.concatenate((X, np.array([x_train]).T), axis=1)

Y = np.array([y_train]).T

theta_best = (np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)).flatten()

print("\nBest Theta: ", theta_best)

# calculate error

MSE = mse(x_test, y_test, theta_best)

print("MSE: ", MSE, "\n")

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# standardization

x_mean = np.mean(x_train)
x_deviation = np.std(x_train)
y_mean = np.mean(y_train)
y_deviation = np.std(y_train)

x_train = (x_train - x_mean)/x_deviation
x_test = (x_test - x_mean)/x_deviation
y_train = (y_train - y_mean)/y_deviation
y_test = (y_test - y_mean)/y_deviation

print("-- After Serialization --")

# calculate theta using Batch Gradient Descent

X = np.ones((m, 1))
X = np.concatenate((X, np.array([x_train]).T), axis=1)

Y = np.array([y_train]).T

n = 0.001

temp_theta = np.array([[random.random()], [random.random()]])
temp_theta2 = temp_theta - n * ((2/m)*np.matmul(X.T, (np.matmul(X, temp_theta) - Y)))

while mse(x_train, y_train, temp_theta.flatten()) != mse(x_train, y_train, temp_theta2.flatten()):
    temp_theta = temp_theta2
    temp_theta2 = temp_theta - n * (2/m)*np.matmul(X.T, (np.matmul(X, temp_theta) - Y))

theta_best = temp_theta2.flatten()

print("Best Theta: ", theta_best)

# calculate error

MSE = mse(x_test, y_test, theta_best)

print("MSE: ", MSE)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()