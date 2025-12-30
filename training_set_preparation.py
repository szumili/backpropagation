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


# training set 
def prepare_training_set(self):

    self.zb_uczacy = {}
        
    dataset = [] # training set x
    dataset_y = [] # training set y
    for cyfra in self.numbers_from_mnist:
        print(cyfra)
        for wzor in self.numbers_from_mnist[cyfra]:
            przyklad1 = wzor

            przyklad2 = self.fourier_transform(wzor)
            przyklad = np.concatenate([przyklad1, przyklad2])
            przyklad = np.array(przyklad)

            dataset.append(przyklad) # adding a sample to the training set

            lista = np.zeros(10)
            lista[cyfra] = 1
            print(lista)
            dataset_y.append(list(lista)) # and we check whether it is a drawing of the digit for which we are creating the perceptron
    
    dataset.reverse()
    dataset_y.reverse()
           
    dataset = np.array(dataset)
    dataset_y = np.array(dataset_y)
            
    self.zb_uczacy['x'] = dataset
    self.zb_uczacy['y'] = dataset_y
        
