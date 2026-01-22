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
    
    def forward(self, x):

        self.y0 = x.copy()
        self.a1 = np.dot(self.y0, self.W1) #+ self.b1
        self.y1 = self.sigmoid(self.a1)

        self.a2 = np.dot(self.y1, self.W2) #+ self.b2
        self.y2 = self.sigmoid(self.a2)

        self.a3 = np.dot(self.y2, self.W3) #+ self.b3
        self.y3 = self.sigmoid(self.a3)

        self.a4 = np.dot(self.y3, self.W4) #+ self.b4
        self.y4 = self.sigmoid(self.a4)


    def backward(self, y):

        self.epsilon4 = y - self.y4
        self.delta4 = self.epsilon4 * self.sigmoid_derivative(self.y4)

        self.epsilon3 = self.delta4.dot(self.W4.T)
        self.delta3 = self.epsilon3 * self.sigmoid_derivative(self.y3)

        self.epsilon2 = self.delta3.dot(self.W3.T)
        self.delta2 = self.epsilon2 * self.sigmoid_derivative(self.y2)

        self.epsilon1 = self.delta2.dot(self.W2.T)
        self.delta1 = self.epsilon1 * self.sigmoid_derivative(self.y1)

        self.W4 += self.eta*self.y3.T.dot(self.delta4)
        self.W3 += self.eta*self.y2.T.dot(self.delta3)
        self.W2 += self.eta*self.y1.T.dot(self.delta2)
        self.W1 += self.eta*self.y0.T.dot(self.delta1)


    def save_weights(self):

        save_weights_to_file(self.W1, 'weights1.txt')
        save_weights_to_file(self.W2, 'weights2.txt')
        save_weights_to_file(self.W3, 'weights3.txt')
        save_weights_to_file(self.W4, 'weights4.txt')

    
    def train(self, x, y):
        self.forward(x)
        self.backward(y)
        self.errors.append(self.loss(y, self.y4))

    def loss(self, ty, y):
        return np.mean(np.square(ty-y))
    
    def predict(self, x):
        prediction = self.forward(x)
        return(prediction)
