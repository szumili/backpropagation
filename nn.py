import numpy as np

from weights import load_weights, save_weights_to_file

class Neural_Network():

    def __init__(self, structure, eta=0.001): #0.1):
        assert len(structure) == 5
        self.structure = structure
        self.input_dim = self.structure[0]
        self.input_dim = self.structure[0]
        self.hidden_dim_1 = self.structure[1]
        self.hidden_dim_2 = self.structure[2]
        self.hidden_dim_3 = self.structure[3]
        self.output_dim = self.structure[4]
        self.eta = eta
        self.errors = []

            
        try:
            print('Loading weights from files...')

            self.W1 = load_weights('weights1.txt')
            self.W2 = load_weights('weights2.txt')
            self.W3 = load_weights('weights3.txt')
            self.W4 = load_weights('weights4.txt')

            print('Done')

        except:

            print('Loading random weights...')
            self.W1 = 2*np.random.random((self.input_dim, self.hidden_dim_1 )) - 1
            #self.b1 = 2*np.random.random((self.input_dim, self.hidden_dim )) - 1 #biased
            self.W2 = 2*np.random.random((self.hidden_dim_1, self.hidden_dim_2)) - 1
            self.W3 = 2*np.random.random((self.hidden_dim_2, self.hidden_dim_3)) - 1  
            self.W4 = 2*np.random.random((self.hidden_dim_3, self.output_dim)) - 1   

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, s):
        return s*(1-s)