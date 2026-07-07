import numpy as np

from keras.datasets import mnist
from random import sample 
from tqdm import tqdm

from random import choice

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from training_set_preparation import getting_numbers_from_mnist, fourier_transform, prepare_dataset
from nn import Neural_Network

def one_hot_y_to_label(y):
    return np.argmax(y, axis=1)



if __name__ == "__main__":

    train_dict, test_dict, numbers_from_mnist = getting_numbers_from_mnist()
    #print(train_dict[0][0])
    train_set = prepare_dataset(train_dict)
    test_set = prepare_dataset(test_dict)
    #print(train_set['x'][0], train_set['y'][0])

    network = Neural_Network([2*28*28, 130, 64, 32, 10]) # creating a neural network

    y_label_train = one_hot_y_to_label(train_set['y'])
    y_label_test = one_hot_y_to_label(test_set['y'])

    #print(train_set['x'][0])

    y_pred_train = [network.predict(digit) for digit in tqdm(train_set['x'])]
    y_pred_test = [network.predict(digit) for digit in tqdm(test_set['x'])]

    y_pred_label_train = one_hot_y_to_label(y_pred_train)
    y_pred_label_test = one_hot_y_to_label(y_pred_test)

    #print(len(y_pred_train), y_pred_train[0], len(y_pred_label_train), y_pred_label_train[0])

    print('accuracy - train: ', accuracy_score(y_label_train, y_pred_label_train))
    print('accuracy - test: ', accuracy_score(y_label_test, y_pred_label_test))

    print('train: ', classification_report(y_label_train, y_pred_label_train, labels=list(range(10)), zero_division=0))
    print('test: ', classification_report(y_label_test, y_pred_label_test, labels=list(range(10)), zero_division=0))

    print('train: ', confusion_matrix(y_label_train, y_pred_label_train, labels=list(range(10))))
    print('test: ', confusion_matrix(y_label_test, y_pred_label_test, labels=list(range(10))))



