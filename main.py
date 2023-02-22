# Author: Bobby Bose
# Assignment 3: Multilayer Perceptrons 


# Imports
import math
import numpy as np
import pandas as pd


# Main function of the program and neural net
def main():

    # Reading in training and testing datasets
    training_dataset_df = pd.read_csv("datasets/mnist_train_0_1.csv")
    test_dataset_df = pd.read_csv("datasets/mnist_test_0_1.csv")
    
    # Splitting data into x and y and normalizing
    training_data_x = list(map(lambda x: x/255, training_dataset_df.iloc[:, 1:].to_numpy()))
    training_data_y = list(training_dataset_df.iloc[:, 0])
    test_data_x = list(map(lambda x: x/255, test_dataset_df.iloc[:, 1:].to_numpy()))
    test_data_y = list(test_dataset_df.iloc[:, 0])

    # 784 Input features
    # Structure: 784 -> 50 -> 1
    num_nodes_in_layer = [784, 50, 1]

    # 784x50
    weights_h = []
    # 50x1
    weights_o = []
    
    # Initializing weights
    for i in range(num_nodes_in_layer[0]):
        weights_h.append(list(np.random.uniform(-1, 1, num_nodes_in_layer[1])))

    for i in range(num_nodes_in_layer[1]):
        weights_o.append(np.random.uniform(-1, 1))

    # 50x1
    bias_h = np.random.uniform(0, 1, num_nodes_in_layer[1]).tolist()
    # 1x1
    bias_o = np.random.uniform(0, 1, num_nodes_in_layer[2]).tolist()

    # Parameters for neural net
    alpha = 0.5
    epochs = 500

    # Training neural net
    for epoch in range(epochs):
        for index, x in enumerate(training_data_x):
            # data is [784 x 1]
            weights_h, weights_o, bias_h, bias_o = backpropagation(alpha, weights_h, weights_o, bias_h, bias_o, x, training_data_y[index])
            

    # Track how many predictions are correct
    num_correct = 0
    # Testing neural net
    for index, x in enumerate(test_data_x):
        in_h, h, in_o, prediction = forward_pass(weights_h, weights_o, bias_h, bias_o, x)

        # Rounding the prediction up or down to the class label
        if prediction >= 0.5:
            prediction = 1
        else:
            prediction = 0

        # Tallying up correct predictions
        if prediction == test_data_y[index]:
            num_correct += 1

    # Printing the neural net accuracy
    print("Num correct: ", num_correct)
    print("Neural Net accuracy:", num_correct/len(test_data_y))


# Description: Runs backpropagation through the neural network and updates weights
# Arguments: learning rate, weights and biases of neural net, input, and correct class label
# Returns: Updated weights and biases
def backpropagation(alpha, weights_h, weights_o, bias_h, bias_o, x, y):
    
    # Values needed for updating
    in_h, h, in_o, o = forward_pass(weights_h, weights_o, bias_h, bias_o, x)
    error = y - o
    # 1x1
    delta_o = error * sigmoid_prime(in_o)

    # 50x1
    delta_h = np.multiply( np.dot(weights_o, delta_o), list(map(sigmoid_prime, in_h)))

    # Updating weights and biases
    weights_o = weights_o + alpha * np.dot(h, delta_o)
    bias_o = bias_o + np.array(alpha * delta_o)
    weights_h = weights_h + alpha * np.outer(x, np.array(delta_h).transpose())
    bias_h = bias_h + alpha * delta_h

    return weights_h, weights_o, bias_h, bias_o


# Description: Calculates the forward pass of the neural net
# Arguments: weights and biases of neural net, input
# Returns: Neural Net prediction
def forward_pass(weights_h, weights_o, bias_h, bias_o, x):
    # 50x1
    in_h = np.dot(np.array(weights_h).transpose(), x) + bias_h

    # 50x1
    h = list(map(sigmoid, in_h))

    # 1x1
    in_o = np.dot(np.array(weights_o).transpose(), h) + bias_o

    # 1x1
    o = sigmoid(in_o)

    return in_h, h, in_o, o


# Description: Calculates sigmoid function result of x
# Arguments: x
# Returns: Sigmoid function of x
def sigmoid(x):
    return 1/(1+math.exp(-x))


# Description: Calculates derivative of sigmoid function result of x
# Arguments: x
# Returns: Derivative of sigmoid of x
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))


main()