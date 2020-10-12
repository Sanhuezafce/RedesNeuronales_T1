from RedesNeuronales_T1.NeuralNetwork_Unmodified import *
def main():
# Set the seed to make result reproducible
    np.random.seed(42)

# The 4 training examples by columns
    X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])

# The outputs of the XOR for every example in X
    Y = np.array([[0, 1, 1, 0]])

# No. of training examples
    m = X.shape[1]

# Set the hyperparameters
    n_x = 2     #No. of neurons in first layer
    n_h = 4     #No. of neurons in hidden layer
    n_y = 1     #No. of neurons in output layer

#The number of times the model has to learn the dataset
    number_of_iterations = 10000
    learning_rate = 0.01

# define a model
    trained_parameters = model(X, Y, n_x, n_h, n_y, number_of_iterations, learning_rate,m)

# Test 2X1 vector to calculate the XOR of its elements.
# You can try any of those: (0, 0), (0, 1), (1, 0), (1, 1)
    X_test = np.array([[0], [1]])
    y_predict = predict(X_test, trained_parameters)

# Print the result
    print('Neural Network prediction for example ({:d}, {:d}) is {:d}'.format(
    X_test[0][0], X_test[1][0], y_predict))


if __name__ == "__main__":
    main()
