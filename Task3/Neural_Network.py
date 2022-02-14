import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

train_data = []
test_data = []
w = []                   # weights
input_layer_size = 5     # 4 features + bias
output_layer_size = 3    # 3 classes

output_layer_activation = []
gradiant_layer = []

con_matrix = []
accuracy = 0

Iris_data = pd.read_csv('IrisData.txt')


def Define_Data():
    train_data.clear()
    test_data.clear()

    for i in range(3):
        start_idx = i * 50
        class_list = Iris_data.iloc[start_idx: start_idx + 50]   # split Iris_data to 3 classes

        class_list = class_list.sample(frac=1).reset_index(drop=True)  # Shuffle Rows

        class_name = class_list.iloc[0, 4]
        class_list = class_list.replace(class_name, i)  # Replace Class Name with number(0, 1, 2)

        train_data.append(class_list.iloc[:30, :])      # add to Train Lists first 30 rows
        test_data.append(class_list.iloc[30:, :])       # add to Test Lists Last 20 rows


def add_bias_and_reshape(c, sample, bias):
    new_x = list(c.iloc[sample, :4])      # class C index(sample_index, 4_feature)
    new_x.insert(0, bias)
    new_x = np.array(new_x).reshape(5, 1)  # new_x was considered 1D array [shape = (5,)], so we reshape it to 5x1
    return new_x


def initialize_weights(hidden_layers, neurons_list):
    w.clear()
    
    for i in range(hidden_layers+1):
        if i == 0:
            w_layer = np.random.rand(neurons_list[i], input_layer_size)

        elif i == hidden_layers:
            w_layer = np.random.rand(output_layer_size, neurons_list[i-1])

        else:
            w_layer = np.random.rand(neurons_list[i], neurons_list[i-1])

        w.append(w_layer)


def activation_func(method, net):
    if method == 'Sigmoid':
        return 1 / (1 + np.exp(-net))
    else:
        return (1 - np.exp(-net)) / (1 + np.exp(-net))


def derivative_activation_func(method, layer):
    if method == 'Sigmoid':
        derv_sigmoid = lambda i: i*(1-i)
        vectorized_derivative = np.vectorize(derv_sigmoid)
        result = vectorized_derivative(layer)

    else:
        derv_hyp_tang = lambda i: (1-i)*(1+i)
        vectorized_derivative = np.vectorize(derv_hyp_tang)
        result = vectorized_derivative(layer)

    return result


def check_classification(output_net, target_class):
    res = 0
    max_element = max(output_net)
    index = np.where(output_net == max_element)

    mapping_yPred = np.zeros([3, 1])
    mapping_yPred[index] = 1

    mapping_target = np.zeros([3, 1])
    mapping_target[target_class] = 1

    diff = mapping_target - mapping_yPred
    if all(r != 0 for r in diff):
        res = diff

    return res, index[0][0]


def step_forward(hidden_layers, x, activation_method):
    output_layer_activation.clear()

    for i in range(hidden_layers+1):
        neurons_net = np.dot(w[i], x)

        output_net = activation_func(activation_method, neurons_net)

        output_layer_activation.append(output_net)
        x = output_net


def step_backward(hidden_layers, activation_method, target_index):
    gradiant_layer.clear()

    # calculate Gradient for output layer
    output_net = output_layer_activation[-1]

    diff = check_classification(output_net, target_index)
    if diff == 0:
        return False

    Der_Y = derivative_activation_func(activation_method, output_net)
    gradiant_layer.append(np.multiply(diff[0], Der_Y))

    counter = 1
    for layer in range(hidden_layers):
        transpose_weights = np.transpose(w[-counter])

        first = np.dot(transpose_weights, gradiant_layer[-counter])       # first = w * gradiant
        second = derivative_activation_func(activation_method, output_layer_activation[-counter - 1])   # second = f'(net)

        gradiant_layer.insert(0, np.multiply(first, second))
        counter += 1

    return True


def update_weights(Input, alpha):
    i = 0
    for j in range(len(w)):
        Input_T = np.transpose(Input)

        first = np.dot(gradiant_layer[i], Input_T)           # first = gradiant * x
        second = alpha * first                               # second = alpha * ( gradiant * x )
        w[j] = w[j] + second                             # new_w = w + (alpha * gradiant * x)

        Input = output_layer_activation[i]
        i += 1


def evaluate(pred_class, actual_class):
    global con_matrix
    con_matrix = confusion_matrix(actual_class, pred_class)

    correct = np.trace(con_matrix)
    global accuracy
    accuracy = correct / np.sum(con_matrix) * 100


def data_test(bias, activation_method, hidden_layers):
    pred_class = []
    actual_class = []
    global w
    for c in test_data:
        for sample in range(len(c)):

            x = add_bias_and_reshape(c, sample, bias)
            step_forward(hidden_layers, x, activation_method)
            output_net = output_layer_activation[-1]
            diff, index = check_classification(output_net, c.iloc[sample, 4])
            pred_class.append(index)
            actual_class.append(c.iloc[sample, 4])

    evaluate(pred_class, actual_class)


def back_propagation_training(epochs, learning_rate, hidden_layers, neurons_list, activation_method, bias):

    initialize_weights(hidden_layers, neurons_list)

    for i in range(epochs):
        for c in train_data:             # c = c1, c2, c3
            for sample in range(len(c)):
                x = add_bias_and_reshape(c, sample, bias)
                step_forward(hidden_layers, x, activation_method)
                state = step_backward(hidden_layers, activation_method, c.iloc[sample, 4])
                if state:
                    update_weights(x, learning_rate)


def Run(epochs, learning_rate, hidden_layers, neurons_list, activation_method, bias):
    Define_Data()
    back_propagation_training(epochs, learning_rate, hidden_layers, neurons_list, activation_method, bias)
    data_test(bias, activation_method, hidden_layers)
    print('done')
