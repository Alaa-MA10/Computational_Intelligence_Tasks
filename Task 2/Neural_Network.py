import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

train_data = []
test_data = []
classes_names = []
confusion_matrix = []
accuracy = 0
w = []  # weights

Iris_data = pd.read_csv('IrisData.txt')


def Draw_Iris_dataset():
    Feature_List = [Iris_data['X1'], Iris_data['X2'], Iris_data['X3'], Iris_data['X4']]
    c = 1
    for feature_pair in itertools.combinations(Feature_List, 2):
        plt.figure(c)
        plt.scatter(feature_pair[0][:50], feature_pair[1][:50])
        plt.scatter(feature_pair[0][50:100], feature_pair[1][50:100])
        plt.scatter(feature_pair[0][100:150], feature_pair[1][100:150])

        plt.xlabel(feature_pair[0].name, fontsize=15)
        plt.ylabel(feature_pair[1].name, fontsize=15)

        c += 1
    plt.show(block=False)


def Define_Data(X, C):
    train_data.clear()
    test_data.clear()
    Iris_data_filtered = Iris_data.loc[:, ['X'+str(X[0]), 'X'+str(X[1]), 'Class']]   # select wanted features (X col)

    for i in range(len(C)):
        start_idx = (C[i]-1)*50
        class_list = Iris_data_filtered.iloc[start_idx: start_idx+50]   # split Iris_data to wanted classes

        class_list = class_list.sample(frac=1).reset_index(drop=True)  # Shuffle Rows
        class_name = class_list.iloc[0, 2]
        classes_names.append(class_name)
        class_list = class_list.replace(class_name, (-2*i)+1)     # Replace Class Name with number(-1, 1)

        train_data.append(class_list.iloc[:30, :])      # add to Train Lists first 30 rows
        test_data.append(class_list.iloc[30:, :])       # add to Test Lists Last 20 rows


def Signum(WtX):
    if WtX > 0:
        return 1
    return -1


def add_bias_and_reshape(l, c, bias):
    L = list(l.iloc[c, :2])
    L.insert(0, bias)
    L = np.array(L).reshape(3, 1)  # L was considered 1D array [shape = (3,)], so we reshape it to 3x1
    return L


def perceptron_training(epochs, bias, learning_rate):
    global w
    w = np.random.rand(1, 3)        # w is transposed, original: 3x1, w is now 1x3
    for i in range(epochs):
        for j in train_data:
            for c in range(len(j)):
                x = add_bias_and_reshape(j, c, bias)
                yPred = Signum(w.dot(x))
                if yPred != j.iloc[c, 2]:
                    Loss = j.iloc[c, 2] - yPred
                    x = np.transpose(x)    # x now is 1x3 to use it in sum (should be same size as w)
                    w = sum(w, learning_rate*Loss*x)



def Adaline_training(bias , learning_rate, Mse_therehold):
    global w
    error_sum = 0
    w = np.random.rand(1, 3)
    for i in range(10000):
        for j in train_data:
            for c in range(len(j)):
                x = add_bias_and_reshape(j, c, bias)
                yPred = w.dot(x)
                error = j.iloc[c, 2] - yPred
                x = np.transpose(x)
                w = sum(w, learning_rate*error*x)
        for j in train_data:
            for c in range(len(j)):
                x = add_bias_and_reshape(j, c, bias)
                yPred = w.dot(x)
                error = j.iloc[c, 2] - yPred[0][0]
                error_sum += (error * error)/2
        MSE = error_sum/ len(train_data[0]*2)
        if MSE < Mse_therehold:
            break
        else:
            continue


def evaluate(correct_list):
    testing_elements_num = len(test_data[0])
    global confusion_matrix
    confusion_matrix = np.array([[correct_list[0], testing_elements_num - correct_list[0]],
                                 [testing_elements_num - correct_list[1], correct_list[1]]])
    total_correct = sum(correct_list)
    global accuracy
    accuracy = total_correct / np.sum(confusion_matrix) * 100


def data_test(bias):
    correct_list = []
    global w
    for j in test_data:
        correct_counter = 0
        for c in range(len(j)):
            x = add_bias_and_reshape(j, c, bias)
            yPred = Signum(w.dot(x))
            if yPred == j.iloc[c, 2]:
                correct_counter += 1
        correct_list.append(correct_counter)
    evaluate(correct_list)


def draw_line(X, C, bias):
    # drawing line... equation is w0*x0 + w1*x1 + w2*x2 = 0 --> w0*x0 = b
    plt.figure()
    plt.scatter(test_data[0].iloc[:, 0], test_data[0].iloc[:, 1])
    plt.scatter(test_data[1].iloc[:, 0], test_data[1].iloc[:, 1])

    # getting max number for feature 1 in 2 classes
    first_class_first_x_max = max(test_data[0].iloc[:, 0])
    second_class_first_x_max = max(test_data[1].iloc[:, 0])
    x_max = max(first_class_first_x_max, second_class_first_x_max)

    # getting min number for feature 1 in 2 classes
    first_class_first_x_min = min(test_data[0].iloc[:, 0])
    second_class_first_x_min = min(test_data[1].iloc[:, 0])
    x_min = min(first_class_first_x_min, second_class_first_x_min)

    x1 = np.linspace(x_min, x_max)
    plt.plot(x1, (-w[0][0]-(w[0][1]*x1))/w[0][2], 'k-', label="Accuracy")

    plt.title(str(classes_names[0] + " , " + str(classes_names[1])))
    plt.xlabel("X" + str(X[0]))
    plt.ylabel("X" + str(X[1]))
    plt.show(block=False)


def Learn(X, C, LearningRate, Epochs, Bias , Mse_therehold):
    Define_Data(X, C)
    #perceptron_training(Epochs, Bias, LearningRate)
    Adaline_training(Bias , LearningRate , Mse_therehold)
    data_test(Bias)
    draw_line(X, C, Bias)


Draw_Iris_dataset()
