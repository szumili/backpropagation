import sys

import numpy as np
import matplotlib.pyplot as plt

from random import choice, sample 
from keras.datasets import mnist

from training_set_preparation import getting_numbers_from_mnist, fourier_transform, prepare_training_set

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QMessageBox 
from PyQt5.QtCore import Qt



class Grid(QWidget):

    def __init__(self, width, height, cell_size):
        super().__init__() 
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = [[False for _ in range(width)] for _ in range(height)]
        self.buttons = [[QPushButton(self) for _ in range(width)] for _ in range(height)]


        self.numbers_from_mnist = getting_numbers_from_mnist()
        self.training_set = prepare_training_set(self.numbers_from_mnist)
        self.siec_tworzenie()

        self.drawing = True


        for row in range(height):
            for col in range(width):
                self.buttons[row][col].setStyleSheet(f"background-color: white; border: 1px solid black")
                self.buttons[row][col].setFixedSize(cell_size, cell_size)
                self.buttons[row][col].clicked.connect(self.make_toggle(row, col))


        mainLayout = QHBoxLayout()

        # grid
        self.layout = QGridLayout()
        self.layout.setSpacing(0)
        for row in range(height):
            for col in range(width):
                self.layout.addWidget(self.buttons[row][col], row, col)


        mainLayout.addLayout(self.layout)



        self.layoutButtons = QVBoxLayout() # layout with my buttons
        mainLayout.addLayout(self.layoutButtons)

        # show the matrix
        buttonMatrix = QPushButton('Show the matrix')
        buttonMatrix.clicked.connect(self.matrix)
        self.layoutButtons.addWidget(buttonMatrix)

        # empty
        buttonEmpty = QPushButton('Clear')
        buttonEmpty.clicked.connect(self.empty)
        self.layoutButtons.addWidget(buttonEmpty)

        # training
        buttonUczenie = QPushButton('Train')
        buttonUczenie.clicked.connect(self.siec_uczenie)
        self.layoutButtons.addWidget(buttonUczenie)

        # guessing
        buttonWhich = QPushButton('What digit is this?')
        buttonWhich.clicked.connect(self.zgadywanie)
        self.layoutButtons.addWidget(buttonWhich)




        # noise
        buttonSzum = QPushButton('Add noise')
        buttonSzum.clicked.connect(self.szum)
        self.layoutButtons.addWidget(buttonSzum)



        # negative
        buttonNeg = QPushButton('Negative')
        buttonNeg.clicked.connect(self.neg)
        self.layoutButtons.addWidget(buttonNeg)





        # up
        buttonUp = QPushButton('↑')
        buttonUp.clicked.connect(self.up)
        self.layoutButtons.addWidget(buttonUp)

        
        # down
        buttonDown = QPushButton('↓')
        buttonDown.clicked.connect(self.down)
        self.layoutButtons.addWidget(buttonDown)

        
        # left
        buttonLeft = QPushButton('←')
        buttonLeft.clicked.connect(self.left)
        self.layoutButtons.addWidget(buttonLeft)

        
        # right
        buttonRight = QPushButton('→')
        buttonRight.clicked.connect(self.right)
        self.layoutButtons.addWidget(buttonRight)

        
        # 0
        buttonZero = QPushButton('Draw 0')
        buttonZero.clicked.connect(lambda: self.drawNumber(0))
        self.layoutButtons.addWidget(buttonZero)

        # 1 
        buttonOne = QPushButton('Draw 1')
        buttonOne.clicked.connect(lambda: self.drawNumber(1))
        self.layoutButtons.addWidget(buttonOne)

        # 2
        buttonTwo = QPushButton('Draw 2')
        buttonTwo.clicked.connect(lambda: self.drawNumber(2))
        self.layoutButtons.addWidget(buttonTwo)

        # 3
        buttonThree = QPushButton('Draw 3')
        buttonThree.clicked.connect(lambda: self.drawNumber(3))
        self.layoutButtons.addWidget(buttonThree)

        # 4
        buttonFour = QPushButton('Draw 4')
        buttonFour.clicked.connect(lambda: self.drawNumber(4))
        self.layoutButtons.addWidget(buttonFour)

        # 5
        buttonFive = QPushButton('Draw 5')
        buttonFive.clicked.connect(lambda: self.drawNumber(5))
        self.layoutButtons.addWidget(buttonFive)

        # 6
        buttonSix = QPushButton('Draw 6')
        buttonSix.clicked.connect(lambda: self.drawNumber(6))
        self.layoutButtons.addWidget(buttonSix)

        # 7
        buttonSeven = QPushButton('Draw 7')
        buttonSeven.clicked.connect(lambda: self.drawNumber(7))
        self.layoutButtons.addWidget(buttonSeven)

        # 8
        buttonEight = QPushButton('Draw 8')
        buttonEight.clicked.connect(lambda: self.drawNumber(8))
        self.layoutButtons.addWidget(buttonEight)

        # 9
        buttonNine = QPushButton('Draw 9')
        buttonNine.clicked.connect(lambda: self.drawNumber(9))
        self.layoutButtons.addWidget(buttonNine)


        


        self.setLayout(mainLayout)
        self.show()



        


    # changing colour when clicked
    def make_toggle(self, row, col):
        def toggle():
            self.grid[row][col] = not self.grid[row][col]
            color = 'black' if self.grid[row][col] else 'white'
            self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black")
        return toggle
    
    

    # drawing by hand 
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()


    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:

            for el in self.whichPixel():
                try:
                    if el[0]>=0 and el[1]>=0:
                        self.grid[el[0]][el[1]] = True
                        color = "black"
                        self.buttons[el[0]][el[1]].setStyleSheet(f"background-color: {color}; border: 1px solid black")
                except:
                    blad = True

            self.lastPoint = event.pos()
            self.update()


    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False


    def whichPixel(self):
        # y - row, x - column
        try:
            podstawowa = [[(self.lastPoint.y())//self.cell_size, (self.lastPoint.x())//self.cell_size]]
        except:
            podstawowa = [[-1, -1]]
        

        # to make the line thicker
        pstwo = np.random.uniform(0,1) 
        if pstwo < 1/2:
            podstawowa.append([podstawowa[0][0], podstawowa[0][1]-1])
        elif pstwo > 1/2:
            podstawowa.append([podstawowa[0][0]-1, podstawowa[0][1]])

        return podstawowa




    # show the matrix
    def matrix(self):
        matrix = []
        for row in range(self.height):
            rows = []
            for col in range(self.width):
                rows.append(self.grid[row][col])
            matrix.append(rows)
        print(matrix) # terminal

        # show in messege box as 0 and 1
        m = []
        for el in matrix:
            l = []
            for e in el:
                l.append(int(e)) # changing into 0, 1
            m.append(' '.join(str(l))) # joining columns using spaces
        macierz = '\n'.join(m) # joining rows using \n 
        QMessageBox.about(self, "Matrix", macierz)


    # cleaning the matrix
    def empty(self):
        for row in range(self.height):
            for col in range(self.width):
                self.grid[row][col] = False # changing all values to false (white)
                self.buttons[row][col].setStyleSheet(f"background-color: white; border: 1px solid black") # changing colour to white



    # adding noise to an image
    def szum(self):
        matrix = []
        for row in range(self.height):
            rows = []
            for col in range(self.width):
                rows.append(self.grid[row][col])
            matrix.append(rows)

        ile_zaburzonych = 0

        for row in range(self.height):
            for col in range(self.width):

                # adding noise with a probability of 1/750
                pstwo = np.random.uniform(0,1)
                if pstwo < 1/750:
                    matrix[row][col] = not matrix[row][col]
                    ile_zaburzonych += 1

                self.grid[row][col] = matrix[row][col] # required to have correct True/False values
                color = 'black' if matrix[row][col] else 'white' # which element should have which colour
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") # colouring with the correct colour
        print('narysowane')
        print('zaszumiane: ', round(((ile_zaburzonych/(self.height*self.width))*100), 2), '%')





    # negative
    def neg(self):
        matrix = []
        for row in range(self.height):
            for col in range(self.width):
                matrix.append(int(self.grid[row][col]))
        print(matrix)

        cyfra = [1-x for x in matrix]
        print(cyfra)

        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) # required to have correct True/False values
                color = 'black' if bool(cyfra[nr_el]) else 'white' # which element should have which colour
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") # colouring with the correct colour




    def siec_tworzenie(self):

        self.siec = self.Neural_Network([2*self.width*self.height, 130, 64, 32, 10]) # creating a neural network
        return self.siec

    
    def siec_uczenie(self):

        for i in range(1000): 
            self.siec.train(self.training_set['x'], self.training_set['y']) # training the network on the training set

        print('Nauczono')

        self.siec.save_weights()
        print('Zapisano')




    # guessing
    def zgadywanie(self):
            
        try: # if already trained

            macierz = []
            for row in range(self.height):
                for col in range(self.width):
                    macierz.append(int(self.grid[row][col]))
            print(macierz) # terminal

            
            tak = []
            nie = []
            score = {}
            score_2 = {}
            self.scores = {}

            przyklad1 = macierz

            przyklad2 = fourier_transform(macierz)
            przyklad = np.concatenate([przyklad1, przyklad2])
            matrix = np.array(przyklad)

            wynik = self.siec.predict(matrix) # predicting whether the matrix represents a given digit
            wwynik = wynik
            for i in range(len(wwynik)):
                w = wwynik[i]
                self.scores[i] = w
                if w >= 1/2:
                    tak.append(i)
                    score[i] = w
                else:
                    nie.append(i)
                    score_2[i] = w

            print(tak)
            print(nie)
            print(score)
            print(score_2)

            score = self.scores.copy()

            max_score = 0
            najprawdopodobniej = '?'
            for nr in score:
                if score[nr] > max_score:
                    max_score = score[nr]
                    najprawdopodobniej = str(nr)

                

            naj_z_podpisem = ['Najprawdopodobniej jest to cyfra: ', najprawdopodobniej]
            naj = '\n'.join(naj_z_podpisem)

            
            t = (' '.join(str(tak))) # joining using spaces
            t_z_podpisem = ['Może to być cyfra: ', t]
            yes = '\n'.join(t_z_podpisem) #  joining using \n (t i podpis)
            print(yes)

            n = (' '.join(str(nie))) # joining using spaces
            n_z_podpisem = ['Nie jest to raczej cyfra: ', n]
            no = '\n'.join(n_z_podpisem) #  joining using \n (n i podpis)
            print(no)

            

            odp = []
            odp.append(naj)
            wyniki = '\n'.join(odp)

            self.histogram()

            QMessageBox.about(self, "Wyniki", wyniki)


            

        except: 
            print('error')



    def histogram(self):
        try:
            plt.figure(1)
            plt.clf() # clear

            names = list(self.scores.keys())
            values = list(self.scores.values())
            plt.bar(range(len(self.scores)), values, tick_label=names)

            plt.show()

        except:
            print('error')


    def heatmap(self):
        data = {}
        for number in range(10):
            # only for 0
            print(np.array_equal(self.numbers_from_mnist[number],self.numbers_from_mnist[0]))
            data[number] = self.siec.lrp(np.array(self.numbers_from_mnist[number]))

            print(len(data))
            print(len(data[number]))
            print(len(data[number][0]))
            d = []
            for j in range(len(data[number][0])):
                suma = 0
                for i in range(len(data[number])):
                    suma += data[number][i][j]
                d.append(suma/len(data[number]))

            
        
            

            przykladowe = []
            print(len(d))
            for row in range(self.height):
                p = []
                for col in range(self.width):
                    p.append(d[row*self.width+col])
                przykladowe.append(p)
            

            nr = number + 2
            plt.figure(nr)
            plt.clf() # clear

            plt.imshow(przykladowe, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title(number)

            plt.show()


    

    # shift

    def up(self):
        matrix = []
        for row in range(self.height):
            rows = []
            for col in range(self.width):
                rows.append(self.grid[row][col])
            matrix.append(rows)
        matrix = np.array(matrix)

        m = np.roll(matrix, -1, axis=0)

        
        for row in range(self.height):
            for col in range(self.width):
                self.grid[row][col] = m[row][col] # required to have correct True/False values
                color = 'black' if m[row][col] else 'white' # which element should have which colour
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") # colouring with the correct colour

    def down(self):
        matrix = []
        for row in range(self.height):
            rows = []
            for col in range(self.width):
                rows.append(self.grid[row][col])
            matrix.append(rows)
        matrix = np.array(matrix)

        m = np.roll(matrix, 1, axis=0)
        
        for row in range(self.height):
            for col in range(self.width):
                self.grid[row][col] = m[row][col] 
                color = 'black' if m[row][col] else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") 

    def left(self):
        matrix = []
        for row in range(self.height):
            rows = []
            for col in range(self.width):
                rows.append(self.grid[row][col])
            matrix.append(rows)
        matrix = np.array(matrix)

        m = np.roll(matrix, -1, axis=1)
        
        for row in range(self.height):
            for col in range(self.width):
                self.grid[row][col] = m[row][col] 
                color = 'black' if m[row][col] else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") 

    def right(self):
        matrix = []
        for row in range(self.height):
            rows = []
            for col in range(self.width):
                rows.append(self.grid[row][col])
            matrix.append(rows)
        matrix = np.array(matrix)

        m = np.roll(matrix, 1, axis=1)
        
        for row in range(self.height):
            for col in range(self.width):
                self.grid[row][col] = m[row][col] 
                color = 'black' if m[row][col] else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black")



 

    def drawNumber(self, digit):
        cyfra = choice(self.numbers_from_mnist[digit]) # picking a sample for a chosen digit
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) # required to have correct True/False values
                color = 'black' if bool(cyfra[nr_el]) else 'white' # which element should have which colour
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") # colouring with the correct colour
                  



    
    class Neural_Network():

        def __init__(self, structure, eta=0.001):#0.1):
            assert len(structure) == 5
            self.structure = structure
            self.input_dim = self.structure[0]
            self.hidden_dim_1 = self.structure[1]
            self.hidden_dim_2 = self.structure[2]
            self.hidden_dim_3 = self.structure[3]
            self.output_dim = self.structure[4]
            self.eta = eta
            self.errors = []

            
            try:
                print('wczytane')

                self.W1 = []
                self.W2 = []
                self.W3 = []
                self.W4 = []

                file1 = open(('wagi1.txt'), 'r')
                file2 = open(('wagi2.txt'), 'r')
                file3 = open(('wagi3.txt'), 'r')
                file4 = open(('wagi4.txt'), 'r')

                data1 = file1.read()
                data1 = data1.strip("\n")
                split_str = data1.split("[")
                for s in split_str[1:]:
                    podlista = []
                    split_s = s.split("]")
                    if len(split_s) > 1:
                        split_s[0] = split_s[0].replace('\n', '')
                        podlista.append(split_s[0].strip('\n').split(" "))
                        podlista2 = []
                        for el in podlista[0]:
                            if len(el)>0:
                                podlista2.append(float(el))
                        self.W1.append(np.array(podlista2))

                data2 = file2.read()
                data2 = data2.strip("\n")
                split_str = data2.split("[")
                for s in split_str[1:]:
                    podlista = []
                    split_s = s.split("]")
                    if len(split_s) > 1:
                        split_s[0] = split_s[0].replace('\n', '')
                        podlista.append(split_s[0].strip('\n').split(" "))
                        podlista2 = []
                        for el in podlista[0]:
                            if len(el)>0:
                                podlista2.append(float(el))
                        self.W2.append(np.array(podlista2))

                data3 = file3.read()
                data3 = data3.strip("\n")
                split_str = data3.split("[")
                for s in split_str[1:]:
                    podlista = []
                    split_s = s.split("]")
                    if len(split_s) > 1:
                        split_s[0] = split_s[0].replace('\n', '')
                        podlista.append(split_s[0].strip('\n').split(" "))
                        podlista2 = []
                        for el in podlista[0]:
                            if len(el)>0:
                                podlista2.append(float(el))
                        self.W3.append(np.array(podlista2))

                data4 = file4.read()
                data4 = data4.strip("\n")
                split_str = data4.split("[")
                for s in split_str[1:]:
                    podlista = []
                    split_s = s.split("]")
                    if len(split_s) > 1:
                        split_s[0] = split_s[0].replace('\n', '')
                        podlista.append(split_s[0].strip('\n').split(" "))
                        podlista2 = []
                        for el in podlista[0]:
                            if len(el)>0:
                                podlista2.append(float(el))
                        self.W4.append(np.array(podlista2))


                self.W1 = np.array(self.W1)
                self.W2 = np.array(self.W2)
                self.W3 = np.array(self.W3)
                self.W4 = np.array(self.W4)
                print(len(self.W4))
                print(len(self.W4[0]))
                
                print('gotowe')

            except:

                print('losowe')
                self.W1 = 2*np.random.random((self.input_dim, self.hidden_dim_1 )) - 1
                #self.b1 = 2*np.random.random((self.input_dim, self.hidden_dim )) - 1 #biased
                self.W2 = 2*np.random.random((self.hidden_dim_1, self.hidden_dim_2)) - 1
                self.W3 = 2*np.random.random((self.hidden_dim_2, self.hidden_dim_3)) - 1  
                self.W4 = 2*np.random.random((self.hidden_dim_3, self.output_dim)) - 1   


        def preprocessing(self, x):
            return 0.8*x+0.1

        def deprocessing(self, y):
            return (y-0.1)/0.8

        def sigmoid(self,x):
            return 1/(1+np.exp(-x))
        
        def sigmoid_derivative(self, s):
            return s*(1-s)
        
        

        def forward(self, x):

            self.y0 = x.copy()
            self.a1 = np.dot(self.y0, self.W1) #+ self.b1
            self.y1 = self.sigmoid(self.a1)

            self.a2 = np.dot(self.y1, self.W2) #+ self.b1
            self.y2 = self.sigmoid(self.a2)

            self.a3 = np.dot(self.y2, self.W3) #+ self.b3
            self.y3 = self.sigmoid(self.a3)

            self.a4 = np.dot(self.y3, self.W4) #+ self.b4
            self.y4 = self.sigmoid(self.a4)

            return self.y4
        
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

            with open(('wagi1.txt'), 'w') as file:
                for el in (self.W1):
                    file.write(str(el))
                    file.write(' ')
                file.write('\n')
                print('zapisano wagi1')

            with open(('wagi2.txt'), 'w') as file:
                for el in (self.W2):
                    file.write(str(el))
                    file.write(' ')
                file.write('\n')
                print('zapisano wagi2')

            with open(('wagi3.txt'), 'w') as file:
                for el in (self.W3):
                    file.write(str(el))
                    file.write(' ')
                file.write('\n')
                print('zapisano wagi3')

            with open(('wagi4.txt'), 'w') as file:
                for el in (self.W4):
                    file.write(str(el))
                    file.write(' ')
                file.write('\n')
                print('zapisano wagi4')
                


        def train(self, x, y):
            self.forward(x)
            self.backward(y)
            self.errors.append(self.loss(y, self.y4))


        def loss(self, ty, y):
            return np.mean(np.square(ty-y))
        

        def predict(self, x):
            wynik = self.forward(x)
            return(wynik)


        def lrp(self, x): # Layer-wise Relevance Propagation
            x = self.preprocessing(x)
            h1 = self.sigmoid(np.dot(x, self.W1))
            h2 = self.sigmoid(np.dot(h1, self.W2))
            h3 = self.sigmoid(np.dot(h2, self.W3))
            y = self.sigmoid(np.dot(h3, self.W4))
            
            R = y * (1 - y)  # output
            R = np.dot(R, self.W4.T)  # 3 warstwa ukryta
            R = R * self.sigmoid_derivative(h3)
            R = np.dot(R, self.W3.T)  # 2 warstwa ukryta
            R = R * self.sigmoid_derivative(h2)
            R = np.dot(R, self.W2.T)  # 1 warstwa ukryta
            R = R * self.sigmoid_derivative(h1)
            R = np.dot(R, self.W1.T)  # input
            R = R * self.sigmoid_derivative(x)
            
            return R
        




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 600, 600) # wspolrzedne polozenia na ekranie (prawo/lewo, gora/dol), szerokosc, wysokosc
        self.setWindowTitle("Zadanie domowe")
        self.grid = Grid(28,28,25)
        self.setCentralWidget(self.grid)
        self.show()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())





