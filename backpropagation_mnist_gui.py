import sys

import numpy as np
import matplotlib.pyplot as plt

from random import choice

from grid import get_matrix
from training_set_preparation import getting_numbers_from_mnist, fourier_transform, prepare_training_set
from nn import Neural_Network

from tqdm import tqdm
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
        self.nn_create()

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
        buttonMatrix.clicked.connect(self.show_matrix)
        self.layoutButtons.addWidget(buttonMatrix)

        # empty
        buttonEmpty = QPushButton('Clear')
        buttonEmpty.clicked.connect(self.empty)
        self.layoutButtons.addWidget(buttonEmpty)

        # training
        buttonTraining = QPushButton('Train')
        buttonTraining.clicked.connect(self.nn_train)
        self.layoutButtons.addWidget(buttonTraining)

        # guessing
        buttonGuess = QPushButton('What digit is this?')
        buttonGuess.clicked.connect(self.guess_digit)
        self.layoutButtons.addWidget(buttonGuess)


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
        buttonUp.clicked.connect(lambda: self.shift_digit(-1, 0))
        self.layoutButtons.addWidget(buttonUp)

        
        # down
        buttonDown = QPushButton('↓')
        buttonDown.clicked.connect(lambda: self.shift_digit(1, 0))
        self.layoutButtons.addWidget(buttonDown)

        
        # left
        buttonLeft = QPushButton('←')
        buttonLeft.clicked.connect(lambda: self.shift_digit(-1, 1))
        self.layoutButtons.addWidget(buttonLeft)

        
        # right
        buttonRight = QPushButton('→')
        buttonRight.clicked.connect(lambda: self.shift_digit(1, 1))
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
    def show_matrix(self):

        matrix = get_matrix(self.height, self.width, self.grid)
        print(matrix) # terminal

        # show in messege box as 0 and 1
        m = [' '.join(str([int(e) for e in el])) for el in matrix] # changing into 0, 1 and joining columns using spaces
        joined_matrix = '\n'.join(m) # joining rows using \n 
        QMessageBox.about(self, "Matrix", joined_matrix)


    # cleaning the matrix
    def empty(self):
        for row in range(self.height):
            for col in range(self.width):
                self.grid[row][col] = False # changing all values to false (white)
                self.buttons[row][col].setStyleSheet(f"background-color: white; border: 1px solid black") # changing colour to white



    # adding noise to an image
    def szum(self):

        matrix = get_matrix(self.height, self.width, self.grid)

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




    def nn_create(self):
        self.network = Neural_Network([2*self.width*self.height, 130, 64, 32, 10]) # creating a neural network
        return self.network

    
    def nn_train(self):

        for i, j in enumerate(tqdm(range(1000), desc="Training the neural network...")): 
            self.network.train(self.training_set['x'], self.training_set['y']) # training the network on the training set

        self.network.save_weights()



    # guessing
    def guess_digit(self):
            
        try: # if already trained
            
            #m = get_matrix(self.height, self.width, self.grid)
            #matrix = [el for row in m for el in row]

            macierz = []
            for row in range(self.height):
                for col in range(self.width):
                    macierz.append(int(self.grid[row][col]))
            print(macierz) # terminal
            #print(macierz == matrix)

            
            tak = []
            nie = []
            score = {}
            score_2 = {}
            self.scores = {}

            przyklad1 = macierz

            przyklad2 = fourier_transform(macierz)
            przyklad = np.concatenate([przyklad1, przyklad2])
            matrix = np.array(przyklad)

            wynik = self.network.predict(matrix) # predicting whether the matrix represents a given digit
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


        except Exception as e:
            print("Error:", e)


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



    # shift

    def shift_digit(self, shift, axis):
        
        matrix = np.array(get_matrix(self.height, self.width, self.grid))

        m = np.roll(matrix, shift, axis=axis)

        for row in range(self.height):
            for col in range(self.width):
                self.grid[row][col] = m[row][col] # required to have correct True/False values
                color = 'black' if m[row][col] else 'white' # which element should have which colour
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") # colouring with the correct colour

 
    def drawNumber(self, digit):
        cyfra = choice(self.numbers_from_mnist[digit]) # picking a sample for a chosen digit
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) # required to have correct True/False values
                color = 'black' if bool(cyfra[nr_el]) else 'white' # which element should have which colour
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") # colouring with the correct colour
                  





class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 600, 600) # wspolrzedne polozenia na ekranie (prawo/lewo, gora/dol), szerokosc, wysokosc
        self.setWindowTitle("Handwritten Digit Recognition")
        self.grid = Grid(28,28,25)
        self.setCentralWidget(self.grid)
        self.show()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

