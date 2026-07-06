import numpy as np

from keras.datasets import mnist
from random import sample 
from tqdm import tqdm



def transform_set(x_set):

    x_set = x_set.reshape(-1, 28*28) / 255.
    x_set[x_set >= 0.5] = 1
    x_set[x_set < 0.5] = 0

    return x_set


def numbers_to_dict(x, y):

    numbers = {}

    for num in range(10):

        indices = np.where(y == num)[0]
        numbers[num] = np.array([x[id] for id in indices])

    return numbers


# downloading data from mnist
def getting_numbers_from_mnist():

    numbers_from_mnist = {}

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(len(x_train), len(x_test))
    print(np.bincount(y_train))
    print(np.bincount(y_test))

    x_train = transform_set(x_train)
    x_test = transform_set(x_test)

    train_dict = numbers_to_dict(x_train, y_train)
    test_dict = numbers_to_dict(x_test, y_test)

    numbers_from_mnist = {k: np.concatenate((train_dict[k], test_dict[k])) for k in train_dict}

    return train_dict, test_dict, numbers_from_mnist


def fourier_transform(x):
    t = np.abs(np.fft.fft(x))
    return ((t-np.mean(t))/np.std(t)) # standardization



# training set 
def prepare_dataset(numbers_from_mnist):

    prepared_dataset = {}
        
    dataset = [] # training set x
    dataset_y = [] # training set y
    for i, digit in enumerate(tqdm(numbers_from_mnist, desc="Loading digits")):
        for digit_pattern in numbers_from_mnist[digit]:

            example1 = digit_pattern

            example2 = fourier_transform(digit_pattern)
            example = np.concatenate([example1, example2])
            example = np.array(example)

            dataset.append(example) # adding a sample to the training set

            digit_list = np.zeros(10)
            digit_list[digit] = 1
            dataset_y.append(list(digit_list)) # we check whether it is a drawing of the digit for which we are creating the perceptron
    
    #dataset.reverse()
    #dataset_y.reverse()
            
    prepared_dataset['x'] = np.array(dataset)
    prepared_dataset['y'] = np.array(dataset_y)

    print(prepared_dataset['x'][0], prepared_dataset['y'][0])

    return prepared_dataset
        


if __name__ == "__main__":
    train_dict, test_dict, numbers_from_mnist = getting_numbers_from_mnist()
    prepare_dataset(numbers_from_mnist)
