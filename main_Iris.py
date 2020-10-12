from pathlib import Path
from RedesNeuronales_T1.T1 import *


def main():
    # Set the seed to make result reproducible
    np.random.seed(42)
    path=Path("iris.data")
    iris_set = data_to_list(path)
    X, Y = decode_data(iris_set)
    iris_encoded = one_hot_encoding(Y)
    Y_int = encode_to_ndarray(Y, iris_encoded)
    m = X.shape[1]

    # Set the hyperparameters
    n_x = 4  # No. of neurons in first layer
    n_h = 4  # No. of neurons in hidden layer
    n_y = 3  # No. of neurons in output layer

    # The number of times the model has to learn the dataset
    number_of_iterations = 10000
    learning_rate = 0.01

    # define a model
    print("Training the parameters . . .")
    trained_parameters, cost_list = iris_model(X, Y_int, n_x, n_h, n_y, number_of_iterations, learning_rate,m)
    print("Parameters trained")
    display_cost(cost_list)
    CM = confusion_matrix(trained_parameters, iris_set)
    kfold(iris_set,3,n_x,n_h,n_y,10000,learning_rate,m)
    some_test = 0
    iris_test = np.array([[X[0][some_test]],
                          [X[1][some_test]],
                          [X[2][some_test]],
                          [X[3][some_test]]])

    actual_result = Y[some_test]
    actual_result = str(iris_encoded[actual_result])
    predicted = str(iris_predict(iris_test, trained_parameters))

    print('Iris Network prediction for example ({:f}, {:f},{:f}, {:f}) is {:s}, the actual result was {:s}'.format(
        iris_test[0][0], iris_test[1][0], iris_test[2][0], iris_test[3][0], predicted, actual_result))


if __name__ == "__main__":
    main()
