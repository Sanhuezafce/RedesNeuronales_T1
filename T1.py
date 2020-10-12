import statistics
import matplotlib.pyplot as plt
from RedesNeuronales_T1 import NeuralNetwork_Unmodified as nn
import numpy as np
from random import shuffle
from prettytable import PrettyTable


# normalizes an input
# input: the values to be normalized (ndarray)
# returns a list of the same length as the input, with all the values normalized.
def normalization(input):
    res = input
    dl = input[0][0]
    dh = input[0][0]
    for x in range(len(input)):
        for y in range(len(input[x])):
            if input[x][y] > dh:
                dl = input[x][y]
            if input[x][y] < dl:
                dl = input[x][y]
    nh = 1
    nl = 0
    for i in range(len(input)):
        for j in range(len(input[i])):
            num = (input[i][j] - dl) * (nh - nl)
            den = (dh - dl)
            X = (num / den) + dl
            input[i][j] = X
    return res


# Create a confusion matrix from a set and a trained network
# network: the trained network with whom we will obtain the predictions
# testing_set: the set of data that the network will use to make predictions
# return: list of vectors that will be the data representation of the printed matrix.
def confusion_matrix(network, testing_set):
    CM_data = []
    X_iris, Y_iris = decode_data(testing_set)
    encoding = one_hot_encoding(Y_iris)
    labels = []
    for i in encoding:
        labels.append(i)

    CM = PrettyTable()
    field_names = []
    field_names.append("Predicted")
    for i in labels:
        field_names.append(i)
    field_names.append("Total_Predicted")
    CM_data.append(field_names)
    CM.field_names = field_names
    result = []
    for n_test in range(len(X_iris[0])):
        features = np.zeros((len(X_iris), 1))
        for f in range(len(X_iris)):
            features[f][0] = (X_iris[f][n_test])
        predicted = iris_predict(features, network)
        result.append([predicted, encoding[Y_iris[n_test]]])

    f = len(labels)
    fields = []
    for i in range(f):
        fields.append(np.zeros(f))
    for test in result:
        col = np.argmax(test[0], 0)
        row = np.argmax(test[1], 0)
        fields[col][row] += 1

    for i in range(len(fields)):
        new_row = []
        new_row.append(labels[i])
        for j in range(len(fields[i])):
            new_row.append(fields[i][j])

        # total predictions of label
        new_row.append(sum(fields[i]))
        CM_data.append(new_row)
        CM.add_row(new_row)

    new_row = []
    new_row.append("Total_By_Label")
    total = np.zeros(f)
    # sumatoria columnas
    for i in range(len(fields)):
        for j in range(len(fields)):
            total[i] += fields[j][i]
    for i in total:
        new_row.append(i)
    new_row.append(len(X_iris[0]))
    CM_data.append(new_row)
    CM.add_row(new_row)

    correct_results = 0
    for i in range(1, len(CM_data) - 1):
        correct_results += CM_data[i][i]

    print(CM)
    print(
        "The percentage of correct predictions: " + str(correct_results / CM_data[len(CM_data) - 1][len(CM_data) - 1]))
    return CM_data


# apply the "one hot encoding" method to the labels in any given set
# training_set: list with all the labels
# return the dictionary with the labels and encoding
def one_hot_encoding(label_list):
    labels = set(label_list)
    labels = list(labels)
    labels.sort()
    n_labels = len(labels)
    dict = {}
    for i in range(len(labels)):
        encoding = np.zeros(n_labels)
        encoding[i] += 1
        dict[labels[i]] = encoding
    return dict


# Convert a .data file to 2 list of features and labels
# features=[[feature1_ex1,feature1_ex2,...,feature1_ex_n],
#          [feature2_ex1,feature2_ex2,...,feature2_ex_n],
#          [    ...   ,    ...     ,...,      ...      ],
#          [feature_n_ex1,feature_n_ex2,...,feature_n_ex_n]]
#
# label  =  [label_ex1,label_ex2,...,label_ex_n]
#
# path: the path of the file to be converted to a list
# returns the ndarray features and label
def decode_data(data_list):
    label = []
    features = np.zeros((len(data_list[0]) - 1, len(data_list)))
    for i in range(len(data_list)):
        for j in range(len(data_list[i])):
            if j == len(data_list[i]) - 1:
                label.append(data_list[i][j])
            else:
                features[j][i] = data_list[i][j]
    features = normalization(features)
    return features, label


# Convert a data file to a list with all the information in the file
# path: the location of the data file
# ret: [[attr,attr,...,attr,label],...,[attr,attr,...,attr,label]]
def data_to_list(path):
    ret = []
    f = open(path, "r")
    contents = f.readlines()
    shuffle(contents)
    for i in range(len(contents)):
        line = (contents[i].split(","))
        line[len(line) - 1] = line[len(line) - 1].rstrip()
        ret.append(line)
    return ret


# This method plot the cost function
# cost_list: a list of all the costs
# return the graphs
def display_cost(cost_list,title="Costo durante el entrenamiento de una red"):
    x = []
    val = 0
    for i in cost_list:
        x.append(val)
        val += 1
    plt.title(title)
    plt.xlabel("Cantidad de epochs realizados")
    plt.ylabel("Coste de la iteracion")
    plt.plot(x, cost_list)
    plt.show()


# apply the kfold cross validation method to a dataset
# dataset: the dataset to be evaluated
# k: number of subsets
# returns a list with statistics obtained from the analisis
def kfold(dataset, k, n_x, n_h, n_y, num_of_iters, learning_rate, m):
    print("Running Kfold Validation . . .")
    shuffle(dataset)
    groups = []
    n_test = len(dataset) // k
    for i in range(k):
        groups.append([])
        for j in range(n_test * (i), n_test * (i + 1)):
            groups[i].append(dataset[j])
    predictions = []
    stats = []
    for i in groups:
        a_prediction = []
        test_data = i
        training_set = []
        for j in groups:
            if j == i:
                pass
            else:
                training_set.extend(j)
        # train model on a training set

        X_iris, Y_iris = decode_data(training_set)

        iris_encoded = one_hot_encoding(Y_iris)
        Y_iris_int = encode_to_ndarray(Y_iris, iris_encoded)

        Iris_parameters, cost_list = iris_model(X_iris, Y_iris_int, n_x, n_h, n_y, num_of_iters, learning_rate, m)
        #display_cost(cost_list,"Costo entrenamiendo Kfold"+"("+str(k)+")")
        #print("CM Kfold: ")
        #CM=confusion_matrix(Iris_parameters,i)

        X_iris, Y_iris = decode_data(test_data)

        for n_test in range(len(X_iris[0])):
            some_test = np.zeros((len(X_iris), 1))
            for feature in range(len(X_iris)):
                some_test[feature][0] = X_iris[feature][n_test]
                if feature == len(X_iris) - 1:
                    a_prediction.append([iris_predict(some_test, Iris_parameters), iris_encoded[Y_iris[n_test]]])
        performance = []
        for i in range(len(a_prediction)):
            predicted = a_prediction[i][0]
            actual = a_prediction[i][1]
            options = len(actual)
            count = 0
            for i in range(options):
                if predicted[i] == actual[i]:
                    count += 1
                else:
                    pass
            if count == 3:
                performance.append(1)
            else:
                performance.append(0)
        predictions.append(performance)
        scores = []
        for set in range(len(predictions)):
            scores.append(0)
            for test in predictions[set]:
                scores[set] += test

        mean = statistics.mean(scores)
        median = statistics.median(scores)
        # std=statistics.stdev(scores)
        stats.append([mean, median])

    mean_list = []
    median_list = []
    for i in range(len(predictions)):
        mean_list.append(stats[i][0])
        median_list.append(stats[i][1])

    mean = statistics.mean(mean_list)
    median = statistics.median(median_list)
    print("The statistics were the following: ")
    print("------------------------------------")
    print("Number of sets: " + str(k))
    print("Total elements in each set:" + str(len(predictions[0])))
    print("Mean: " + str(mean))
    print("Mean as %: "+ str(mean/len(groups[0])))
    print("Median: " + str(median))
    print("Median as %: "+ str(median/len(groups[0])))
    print("-------------------------------------")
    return predictions


# This method takes the list of labels in the dataset and
# the encoding to convert each kind of iris to an ndarray
# labels: the list of labels in the dataset
# coding: the dictionary representing the iris with ndarray
# returns a dictionary with the new coding
def encode_to_ndarray(labels, coding):
    new_labels = np.zeros((len(coding), len(labels)))
    for test in range(len(labels)):
        current_label = coding[labels[test]]
        for option in range(len(coding)):
            new_labels[option][test] = current_label[option]
    return new_labels


# Modified version of the predict method for the iris network
# X: the training set
# parameters: the current parameters in the network
# return: ndarray with a 1 in the position encoding a type of iris
def iris_predict(X, parameters):
    a2, cache = nn.forward_prop(X, parameters)
    yhat = a2
    yhat = np.squeeze(yhat)
    yhat = yhat.tolist()
    # yhat=[a,b,c]
    m_index = yhat.index(max(yhat))
    for i in range(len(yhat)):
        if i == m_index:
            yhat[i] = 1
        else:
            yhat[i] = 0
    yhat = np.array(yhat)
    return yhat


# model is the main function to train a model (modified to return also the cost list)
# X: is the set of training inputs
# Y: is the set of training outputs
# n_x: number of inputs (this value impacts how X is shaped)
# n_h: number of neurons in the hidden layer
# n_y: number of neurons in the output layer (this value impacts how Y is shaped)
def iris_model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate, m):
    parameters = nn.initialize_parameters(n_x, n_h, n_y)
    cost_list = []
    for i in range(0, num_of_iters + 1):
        a2, cache = nn.forward_prop(X, parameters)
        cost = nn.calculate_cost(a2, Y, m)
        cost_list.append(cost)
        grads = nn.backward_prop(X, Y, cache, parameters, m)
        parameters = nn.update_parameters(parameters, grads, learning_rate)
        if (i % 100 == 0):
            pass
            # print('Cost after iteration# {:d}: {:f}'.format(i, cost))

    return parameters, cost_list
