import numpy as np

from random import sample 
from keras.datasets import mnist

# downloading data from mnist
def getting_numbers_from_mnist():

    numbers_from_mnist = {}
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28*28) / 255.
    x_train[x_train >= 0.5] = 1
    x_train[x_train < 0.5] = 0

    numbers = [0,1,2,3,4,5,6,7,8,9]
            
    for num in numbers:

        id_1 = [i for i, x in enumerate(list(y_train)) if x == num]

        # random selection of 100 from each class
        index = sample(id_1, 500)

        lista = []
        for id in index:
            lista.append(x_train[id])
            
        numbers_from_mnist[num] = lista

    return numbers_from_mnist


def fourier_transform(x):
    t = np.abs(np.fft.fft(x))
    return ((t-np.mean(t))/np.std(t)) # standardization



# training set 
def prepare_training_set(numbers_from_mnist):

    training_set = {}
        
    dataset = [] # training set x
    dataset_y = [] # training set y
    for digit in numbers_from_mnist:
        print(digit)
        for digit_pattern in numbers_from_mnist[digit]:

            example1 = digit_pattern

            example2 = fourier_transform(digit_pattern)
            example = np.concatenate([example1, example2])
            example = np.array(example)

            dataset.append(example) # adding a sample to the training set

            digit_list = np.zeros(10)
            digit_list[digit] = 1
            print(digit_list)
            dataset_y.append(list(digit_list)) # we check whether it is a drawing of the digit for which we are creating the perceptron
    
    dataset.reverse()
    dataset_y.reverse()
           
    dataset = np.array(dataset)
    dataset_y = np.array(dataset_y)
            
    training_set['x'] = dataset
    training_set['y'] = dataset_y

    return training_set
        
