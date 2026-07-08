import numpy as np

from keras.datasets import mnist
from random import sample 
from tqdm import tqdm

from random import choice

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from training_set_preparation import getting_numbers_from_mnist, fourier_transform, prepare_dataset
from nn import Neural_Network

def metrics(train_set, test_set, neural_network):

    y_label_train = np.argmax(train_set['y'], axis=1)
    y_label_test = np.argmax(test_set['y'], axis=1)

    y_pred_train = [neural_network.predict(digit) for digit in tqdm(train_set['x'])]
    y_pred_test = [neural_network.predict(digit) for digit in tqdm(test_set['x'])]

    y_pred_label_train = np.argmax(y_pred_train, axis=1)
    y_pred_label_test = np.argmax(y_pred_test, axis=1)

    accuracy_train = accuracy_score(y_label_train, y_pred_label_train)
    accuracy_test = accuracy_score(y_label_test, y_pred_label_test)

    cr_train = classification_report(y_label_train, y_pred_label_train, output_dict=True)
    cr_test = classification_report(y_label_test, y_pred_label_test, output_dict=True)
    
    return accuracy_train, accuracy_test, cr_train, cr_test



if __name__ == "__main__":

    train_dict, test_dict, numbers_from_mnist = getting_numbers_from_mnist()
    train_set = prepare_dataset(train_dict, 'train')
    test_set = prepare_dataset(test_dict, 'test')
    network = Neural_Network([2*28*28, 130, 64, 32, 10])

    print(metrics(train_set, test_set, network))

