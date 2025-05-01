from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout, QMessageBox #, QInputDialog
#from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
import sys
import numpy as np
from random import choice, sample #, randint, shuffle
import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd
from keras.datasets import mnist


class Grid(QWidget):

    def __init__(self, width, height, cell_size):
        super().__init__() #dziedziczenie
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = [[False for _ in range(width)] for _ in range(height)]
        self.buttons = [[QPushButton(self) for _ in range(width)] for _ in range(height)]


        self.getting_numbers_from_mnist()
        self.zbior_uczacy()
        self.siec_tworzenie()


        for row in range(height):
            for col in range(width):
                self.buttons[row][col].setStyleSheet(f"background-color: white; border: 1px solid black")
                self.buttons[row][col].setFixedSize(cell_size, cell_size)
                self.buttons[row][col].clicked.connect(self.make_toggle(row, col))


        mainLayout = QHBoxLayout()

        # grid
        self.layout = QGridLayout()
        self.layout.setSpacing(0)
        #self.layout.setVerticalSpacing(0)
        #self.layout.setHorizontalSpacing(0)
        for row in range(height):
            for col in range(width):
                self.layout.addWidget(self.buttons[row][col], row, col)


        mainLayout.addLayout(self.layout)



        self.layoutButtons = QVBoxLayout() # layout z moimi przyciskami
        mainLayout.addLayout(self.layoutButtons)

        # wyswietlanie jako macierz
        #buttonMatrix = QPushButton('Wyświetl jako macierz')
        buttonMatrix = QPushButton('Wyświetl macierz')
        buttonMatrix.clicked.connect(self.matrix)
        self.layoutButtons.addWidget(buttonMatrix)

        # empty
        buttonEmpty = QPushButton('Wyczyść')
        buttonEmpty.clicked.connect(self.empty)
        self.layoutButtons.addWidget(buttonEmpty)

        # uczenie
        buttonUczenie = QPushButton('Uczenie')
        buttonUczenie.clicked.connect(self.siec_uczenie)
        self.layoutButtons.addWidget(buttonUczenie)

        # zgadywanie
        buttonWhich = QPushButton('Co to za cyfra?')
        buttonWhich.clicked.connect(self.zgadywanie)
        self.layoutButtons.addWidget(buttonWhich)




        # zaszumianie
        buttonSzum = QPushButton('Zaszumianie')
        buttonSzum.clicked.connect(self.szum)
        self.layoutButtons.addWidget(buttonSzum)



        # negatyw
        buttonNeg = QPushButton('Negatyw')
        buttonNeg.clicked.connect(self.neg)
        self.layoutButtons.addWidget(buttonNeg)





        # przesuniecie w gore
        buttonUp = QPushButton('↑')
        buttonUp.clicked.connect(self.up)
        self.layoutButtons.addWidget(buttonUp)

        
        # przesuniecie w dol
        buttonDown = QPushButton('↓')
        buttonDown.clicked.connect(self.down)
        self.layoutButtons.addWidget(buttonDown)

        
        # przesuniecie w lewo
        buttonLeft = QPushButton('←')
        buttonLeft.clicked.connect(self.left)
        self.layoutButtons.addWidget(buttonLeft)

        
        # przesuniecie w prawo
        buttonRight = QPushButton('→')
        buttonRight.clicked.connect(self.right)
        self.layoutButtons.addWidget(buttonRight)

        
        # dla 0
        buttonZero = QPushButton('Narysuj 0')
        buttonZero.clicked.connect(self.drawZero)
        self.layoutButtons.addWidget(buttonZero)

        # dla 1 
        buttonOne = QPushButton('Narysuj 1')
        buttonOne.clicked.connect(self.drawOne)
        self.layoutButtons.addWidget(buttonOne)

        # dla 2
        buttonTwo = QPushButton('Narysuj 2')
        buttonTwo.clicked.connect(self.drawTwo)
        self.layoutButtons.addWidget(buttonTwo)

        # dla 3
        buttonThree = QPushButton('Narysuj 3')
        buttonThree.clicked.connect(self.drawThree)
        self.layoutButtons.addWidget(buttonThree)

        # dla 4
        buttonFour = QPushButton('Narysuj 4')
        buttonFour.clicked.connect(self.drawFour)
        self.layoutButtons.addWidget(buttonFour)

        # dla 5
        buttonFive = QPushButton('Narysuj 5')
        buttonFive.clicked.connect(self.drawFive)
        self.layoutButtons.addWidget(buttonFive)

        # dla 6
        buttonSix = QPushButton('Narysuj 6')
        buttonSix.clicked.connect(self.drawSix)
        self.layoutButtons.addWidget(buttonSix)

        # dla 7
        buttonSeven = QPushButton('Narysuj 7')
        buttonSeven.clicked.connect(self.drawSeven)
        self.layoutButtons.addWidget(buttonSeven)

        # dla 8
        buttonEight = QPushButton('Narysuj 8')
        buttonEight.clicked.connect(self.drawEight)
        self.layoutButtons.addWidget(buttonEight)

        # dla 9
        buttonNine = QPushButton('Narysuj 9')
        buttonNine.clicked.connect(self.drawNine)
        self.layoutButtons.addWidget(buttonNine)


        


        self.setLayout(mainLayout)
        self.show()



        


    # zmiana koloru przy kliknieciu
    def make_toggle(self, row, col):
        def toggle():
            #print(self.grid[row][col])
            self.grid[row][col] = not self.grid[row][col]
            #print(self.grid[row][col])
            color = 'black' if self.grid[row][col] else 'white'
            self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black")
        return toggle
    

    # rysowanie odreczne 
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

            #print(self.lastPoint)   


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
                    #print(blad)

            self.lastPoint = event.pos()
            self.update()


    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False


    def whichPixel(self):
        #print(self.cell_size)
        #print(self.lastPoint.y(), self.lastPoint.x()) # y odpowiada za wiersz a x za kolumne
        podstawowa = [[(self.lastPoint.y())//self.cell_size, (self.lastPoint.x())//self.cell_size]]
        #print(podstawowa)


        # zeby linia byla grubsza
        pstwo = np.random.uniform(0,1) 
        #print(pstwo)
        if pstwo < 1/2:
            podstawowa.append([podstawowa[0][0], podstawowa[0][1]-1])
        elif pstwo > 1/2:
            podstawowa.append([podstawowa[0][0]-1, podstawowa[0][1]])

        #print(podstawowa)
        return podstawowa




    # wyswietlanie macierzy
    def matrix(self):
        matrix = []
        for row in range(self.height):
            rows = []
            for col in range(self.width):
                #print(self.buttons[row][col].styleSheet())
                rows.append(self.grid[row][col])
                #rows.append(int(self.grid[row][col]))
            matrix.append(rows)
        print(matrix) # terminal

        # wyswietlanie w messege box jako 0 i 1
        m = []
        for el in matrix:
            l = []
            for e in el:
                l.append(int(e)) # zamiana na 0, 1
            m.append(' '.join(str(l))) # zlaczenie spacjami kolumn
        macierz = '\n'.join(m) # zlaczenie \n wierszy
        #print(type(macierz))
        QMessageBox.about(self, "Macierz", macierz)


    # czyszczenie macierzy
    def empty(self):
        for row in range(self.height):
            for col in range(self.width):
                self.grid[row][col] = False # zamiana wszystkich wartosci na falsz (bialy)
                self.buttons[row][col].setStyleSheet(f"background-color: white; border: 1px solid black") # zmiana koloru na bialy



    # pobranie danych z mnist
    def getting_numbers_from_mnist(self):

            self.numbers_from_mnist = {}
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            x_train = x_train.reshape(-1, 28*28) / 255.
            #x_test = x_test.reshape(-1, 28*28) / 255.
            x_train[x_train >= 0.5] = 1
            x_train[x_train < 0.5] = 0
            #x_test[x_test >= 0.5] = 1
            #x_test[x_test < 0.5] = 0

            numbers = [0,1,2,3,4,5,6,7,8,9]
            
            for num in numbers:

                id_1 = [i for i, x in enumerate(list(y_train)) if x == num]
                #id_2 = [i for i, x in enumerate(list(y_test)) if x == num]

                # wybor 100 pierwszych (zeby zapisac wagi)
                #index = id_1[:100]

                # wybor losowy 100 z kazdej klasy
                #index = sample(id_1, 100)
                index = sample(id_1, 500)
                #index = id_1

                # gdy wybieramy z train i z test
                #ile_z_train = randint(50, 80) 
                #id_train = sample(id_1, ile_z_train)
                #id_test = sample(id_2, 100-ile_z_train)

                lista = []
                for id in index:
                    lista.append(x_train[id])
            
                self.numbers_from_mnist[num] = lista

            #for i in self.numbers_from_mnist:
                #print(i, ': ', len(self.numbers_from_mnist[i])) 


        
    def fourier_transform(self, x):
        t = np.abs(np.fft.fft(x))
        #return (t-np.mean(t))/max(t)
        #return (t-min(t))/(max(t)-min(t)) # normalizacja
        return ((t-np.mean(t))/np.std(t))

    
    # zbior uczacy 
    def zbior_uczacy(self):

        self.zb_uczacy = {}
        
        dataset = [] # zb uczacy x
        dataset_y = [] # zb uczacy y
        for cyfra in self.numbers_from_mnist:
            print(cyfra)
            for wzor in self.numbers_from_mnist[cyfra]:
                #print(wzor)
                przyklad1 = wzor

                przyklad2 = self.fourier_transform(wzor)
                przyklad = np.concatenate([przyklad1, przyklad2])
                przyklad = np.array(przyklad)

                #przyklad = np.array(przyklad1)

                #print(przyklad)

                dataset.append(przyklad) # dodajemy przyklad do zb uczacego

                lista = np.zeros(10)
                lista[cyfra] = 1
                print(lista)
                dataset_y.append(list(lista)) # i sprawdzamy czy to rysunek cyfry dla ktorej tworzymy prerceptron
    
        dataset.reverse()
        dataset_y.reverse()
           
        dataset = np.array(dataset)
        dataset_y = np.array(dataset_y)
            
        self.zb_uczacy['x'] = dataset
        self.zb_uczacy['y'] = dataset_y
        

        #return self.numbers_from_mnist, self.zb_uczacy


 



    # zaszumienie obrazka
    def szum(self):
        matrix = []
        for row in range(self.height):
            rows = []
            for col in range(self.width):
                rows.append(self.grid[row][col])
            matrix.append(rows)
        #print(matrix) # terminal

        ile_zaburzonych = 0

        for row in range(self.height):
            for col in range(self.width):

                # zaburzanie z pstwem 1/750
                pstwo = np.random.uniform(0,1)
                if pstwo < 1/750:
                    matrix[row][col] = not matrix[row][col]
                    ile_zaburzonych += 1

                self.grid[row][col] = matrix[row][col] # potzrzebne zeby byly prawidlowe wartosci True/False
                color = 'black' if matrix[row][col] else 'white' # zapisujemy co ma byc w jakim kolorze
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") # 'pokolorowanie' na wlasciwy kolor
        print('narysowane')
        print('zaszumiane: ', round(((ile_zaburzonych/(self.height*self.width))*100), 2), '%')





    # negatyw
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
                self.grid[row][col] = bool(cyfra[nr_el]) # potzrzebne zeby byly prawidlowe wartosci True/False
                color = 'black' if bool(cyfra[nr_el]) else 'white' # zapisujemy co ma byc w jakim kolorze
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") # 'pokolorowanie' na wlasciwy kolor




    def siec_tworzenie(self):

        self.siec = self.Neural_Network([2*self.width*self.height, 130, 64, 32, 10]) # tworzymy siec
        #self.siec = self.Neural_Network([self.width*self.height, 130, 64, 32, 10])
        return self.siec

    
    def siec_uczenie(self):

        for i in range(1000): 
            self.siec.train(self.zb_uczacy['x'], self.zb_uczacy['y']) # uczymy siec na zb uczacym

        print('Nauczono')

        self.siec.save_weights()
        print('Zapisano')

        #self.heatmap()



    # zgadywanie
    def zgadywanie(self):
            
        try: #jesli juz nauczony

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

            przyklad2 = self.fourier_transform(macierz)
            przyklad = np.concatenate([przyklad1, przyklad2])
            matrix = np.array(przyklad)

            #matrix = np.array(przyklad1)
            #print(matrix)
            wynik = self.siec.predict(matrix) # przewidywanie czy macierz jest dana cyfra
            #print('oto wynik', wynik)
            #wwynik = self.siec.deprocessing(wynik) 
            wwynik = wynik
            #print(wwynik)
            for i in range(len(wwynik)):
                w = wwynik[i]
                self.scores[i] = w
                #if w >= 0:
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
            #print(najprawdopodobniej)

                

            naj_z_podpisem = ['Najprawdopodobniej jest to cyfra: ', najprawdopodobniej]
            naj = '\n'.join(naj_z_podpisem)

            
            t = (' '.join(str(tak))) # zlaczenie spacjami 
            t_z_podpisem = ['Może to być cyfra: ', t]
            yes = '\n'.join(t_z_podpisem) # zlaczenie \n (t i podpis)
            print(yes)

            n = (' '.join(str(nie))) # zlaczenie spacjami 
            n_z_podpisem = ['Nie jest to raczej cyfra: ', n]
            no = '\n'.join(n_z_podpisem) # zlaczenie \n (n i podpis)
            print(no)

            

            odp = []
            odp.append(naj)
            #odp.append(yes)
            #odp.append(no)
            wyniki = '\n'.join(odp)
            #print(wyniki)

            self.histogram()

            QMessageBox.about(self, "Wyniki", wyniki)


            

        except: 
            print('błąd')



    def histogram(self):
        try:
            plt.figure(1)
            plt.clf() #czyszczenie

            names = list(self.scores.keys())
            values = list(self.scores.values())
            plt.bar(range(len(self.scores)), values, tick_label=names)

            plt.show()

        except:
            print('błąd')


    def heatmap(self):
        data = {}
        for number in range(10):
            # tylko dla 0
            #print(type(self.numbers_from_mnist[0]))
            #data = self.siec.lrp(np.array(self.numbers_from_mnist[0]))
            print(np.array_equal(self.numbers_from_mnist[number],self.numbers_from_mnist[0]))
            data[number] = self.siec.lrp(np.array(self.numbers_from_mnist[number]))
            #data = [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[[1,2,3],[1,2,3],1,2,3]]

            #print(len(data))
            #print(data)
            #print(np.sum(data))
            #print(data[0])

            '''
            print(len(data[0]))
            print(np.sum(data[0]))
            print(np.sum(data[1]))
            print(np.sum(data[2]))
            print(np.sum(data[3]))
            print(np.sum(data[4]))
            '''
            print(len(data))
            print(len(data[number]))
            print(len(data[number][0]))
            d = []
            for j in range(len(data[number][0])):
                suma = 0
                #print(j)
                for i in range(len(data[number])):
                    #print(i)
                    suma += data[number][i][j]
                    #print('suma', suma)
                #print('srednia', j, ':', suma/len(data))
                d.append(suma/len(data[number]))

            
            
            
            #d = data[10]
            #print(len(d[:self.width*self.height]))
            #d = d[:self.width*self.height]
            

            przykladowe = []
            print(len(d))
            for row in range(self.height):
                p = []
                for col in range(self.width):
                    p.append(d[row*self.width+col])
                przykladowe.append(p)
            #print(przykladowe)
            

            nr = number + 2
            plt.figure(nr)
            plt.clf() #czyszczenie

            plt.imshow(przykladowe, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title(number)
            #plt.xlabel("Feature")
            #plt.ylabel("Sample")

            plt.show()

            
        #print(data) 

    

    # przesuniecia

    def up(self):
        matrix = []
        for row in range(self.height):
            rows = []
            for col in range(self.width):
                rows.append(self.grid[row][col])
            matrix.append(rows)
        matrix = np.array(matrix)
        #print(matrix) # terminal


        m = np.roll(matrix, -1, axis=0)

        #print(m)
        
        for row in range(self.height):
            for col in range(self.width):
                self.grid[row][col] = m[row][col] # potzrzebne zeby byly prawidlowe wartosci True/False
                color = 'black' if m[row][col] else 'white' # zapisujemy co ma byc w jakim kolorze
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") # 'pokolorowanie' na wlasciwy kolor
        #print(self.grid)

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



 
            

    
    def drawZero(self):
        #print(self.numbers[0])
        cyfra = choice(self.numbers_from_mnist[0]) # wybor wzoru dla konkretnej cyfry 
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                #print(cyfra[nr_el])
                self.grid[row][col] = bool(cyfra[nr_el]) # potzrzebne zeby byly prawidlowe wartosci True/False
                color = 'black' if bool(cyfra[nr_el]) else 'white' # zapisujemy co ma byc w jakim kolorze
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") # 'pokolorowanie' na wlasciwy kolor
          

    def drawOne(self):
        cyfra = choice(self.numbers_from_mnist[1])  
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) 
                color = 'black' if bool(cyfra[nr_el]) else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") 
        
    def drawTwo(self):
        cyfra = choice(self.numbers_from_mnist[2])  
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) 
                color = 'black' if bool(cyfra[nr_el]) else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") 
        
    def drawThree(self):
        cyfra = choice(self.numbers_from_mnist[3])  
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) 
                color = 'black' if bool(cyfra[nr_el]) else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") 
        
    def drawFour(self):
        cyfra = choice(self.numbers_from_mnist[4])  
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) 
                color = 'black' if bool(cyfra[nr_el]) else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") 
        
    def drawFive(self):
        cyfra = choice(self.numbers_from_mnist[5])  
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) 
                color = 'black' if bool(cyfra[nr_el]) else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") 
        
    def drawSix(self):
        cyfra = choice(self.numbers_from_mnist[6])  
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) 
                color = 'black' if bool(cyfra[nr_el]) else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") 
        
    def drawSeven(self):
        cyfra = choice(self.numbers_from_mnist[7])  
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) 
                color = 'black' if bool(cyfra[nr_el]) else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") 
        
    def drawEight(self):
        cyfra = choice(self.numbers_from_mnist[8])  
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) 
                color = 'black' if bool(cyfra[nr_el]) else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") 
        
    def drawNine(self):
        cyfra = choice(self.numbers_from_mnist[9])  
        print(cyfra)
        for row in range(self.height):
            for col in range(self.width):
                nr_el = row*self.width+col
                self.grid[row][col] = bool(cyfra[nr_el]) 
                color = 'black' if bool(cyfra[nr_el]) else 'white' 
                self.buttons[row][col].setStyleSheet(f"background-color: {color}; border: 1px solid black") 
        
    












    
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

            '''

            try:
                print('gdzie')
                self.W1 = []
                self.W2 = []
                self.W3 = []
                self.W4 = []

                file1 = open(('wagi1.txt'), 'r')
                file2 = open(('wagi2.txt'), 'r')
                file3 = open(('wagi3.txt'), 'r')
                file4 = open(('wagi4.txt'), 'r')

                print('jest')

                data1 = file1.read()
                #print(data1)
                data1 = data1.strip('\n').split(" ")
            
                print(data1)
                substrings = []
                in_brackets = False
                current_substring = []
                for c in data1:
                    if c == "[":
                        in_brackets = True
                    elif c == "]" and in_brackets:
                        substrings.append(current_substring)
                        current_substring = ""
                        in_brackets = False
                    elif in_brackets:
                        current_substring.append(c)
                
                #if current_substring:
                #    substrings.append(current_substring)
                
                print("The element between brackets : " , substrings)
                print('oooooo')

                lines = file1.readlines()
                for line in lines:
                    #print(line)
                    print('blad')
                    if len(line)>1: #ignorowanie gdy pusta linia
                        line = line.strip()
                        #print(line)
                        self.W1.append([int(s) for s in line.split(' ')])

                print('blad2')


                lines = file2.readlines()
                for line in lines:
                    print(len(line))
                    if len(line)>1: #ignorowanie gdy pusta linia
                        line = line.strip()
                        self.W2.append([int(s) for s in line.split(' ')])

                lines = file3.readlines()
                for line in lines:
                    print(len(line))
                    if len(line)>1: #ignorowanie gdy pusta linia
                        line = line.strip()
                        self.W3.append([int(s) for s in line.split(' ')])

                lines = file4.readlines()
                for line in lines:
                    print(len(line))
                    if len(line)>1: #ignorowanie gdy pusta linia
                        line = line.strip()
                        self.W4.append([int(s) for s in line.split(' ')])


                #print('self.W1')
            

                file1.close()
                file2.close()
                file3.close()
                file4.close()

            except:
            '''
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
                        #print(podlista)
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
                        #print(podlista)
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
                        #print(podlista)
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
                        #print(podlista)
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

                #print(self.W4)
                #print(2*np.random.random((self.hidden_dim_3, self.output_dim)) - 1 )
                
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
        
        #def relu(self, x):
            '''
            w = []
            for el in x:
                l = []
                for e in el:
                    l.append(max(e, 0))
                l = np.array(l)
                w.append(l)
            w = np.array(w)
            return w
            '''
            #return np.maximum(0, x)
        
        
        #def relu_derivative(self, x):
            '''
            # Relu Derivative is 1 for x >= 0 and 0 for x < 0
            # https://stackoverflow.com/questions/42042561/relu-derivative-in-backpropagation
            return (x>=0)
            '''
            #return np.where(x <= 0, 0, 1)
        

        def forward(self, x):
            #x = self.preprocessing(x)

            self.y0 = x.copy()
            self.a1 = np.dot(self.y0, self.W1) #+ self.b1
            self.y1 = self.sigmoid(self.a1)
            #self.y1 = self.relu(self.a1)

            self.a2 = np.dot(self.y1, self.W2) #+ self.b1
            self.y2 = self.sigmoid(self.a2)
            #self.y2 = self.relu(self.a2)

            self.a3 = np.dot(self.y2, self.W3) #+ self.b3
            self.y3 = self.sigmoid(self.a3)
            #self.y3 = self.relu(self.a3)

            self.a4 = np.dot(self.y3, self.W4) #+ self.b4
            self.y4 = self.sigmoid(self.a4)
            #self.y4 = self.relu(self.a4)

            return self.y4
        
        def backward(self, y):
            #y = self.preprocessing(y)

            self.epsilon4 = y - self.y4
            self.delta4 = self.epsilon4 * self.sigmoid_derivative(self.y4) 
            #self.delta4 = self.epsilon4 * self.relu_derivative(self.y4)

            self.epsilon3 = self.delta4.dot(self.W4.T)
            self.delta3 = self.epsilon3 * self.sigmoid_derivative(self.y3)
            #self.delta3 = self.epsilon3 * self.relu_derivative(self.y3)

            self.epsilon2 = self.delta3.dot(self.W3.T)
            self.delta2 = self.epsilon2 * self.sigmoid_derivative(self.y2)
            #self.delta2 = self.epsilon2 * self.relu_derivative(self.y2)

            self.epsilon1 = self.delta2.dot(self.W2.T)
            self.delta1 = self.epsilon1 * self.sigmoid_derivative(self.y1)
            #self.delta1 = self.epsilon1 * self.relu_derivative(self.y1)

            self.W4 += self.eta*self.y3.T.dot(self.delta4)
            self.W3 += self.eta*self.y2.T.dot(self.delta3)
            self.W2 += self.eta*self.y1.T.dot(self.delta2)
            self.W1 += self.eta*self.y0.T.dot(self.delta1)

            #print(self.W4, '\n')


        #'''

        def save_weights(self):

            #print(self.W1)

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
                
        #'''


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
        
            '''
            self.forward(data)
            print(np.sum(self.y4))
            relevance = [None] * len(self.structure)
            relevance[-1] = self.y4

            for i in reversed(range(len(self.structure)-1)):
                layer = self.structure[i]
                if i == 0:
                    relevance_score = np.dot(relevance[i+1], self.W1.T)
                    relevance[i] = relevance_score * self.sigmoid_derivative(self.y0)
                elif i == 1:
                    relevance_score = np.dot(relevance[i+1], self.W2.T)
                    relevance[i] = relevance_score * self.sigmoid_derivative(self.y1)
                elif i == 2:
                    relevance_score = np.dot(relevance[i+1], self.W3.T)
                    relevance[i] = relevance_score * self.sigmoid_derivative(self.y2)
                elif i == 3:
                    relevance_score = np.dot(relevance[i+1], self.W4.T)
                    relevance[i] = relevance_score * self.sigmoid_derivative(self.y3)

            return relevance
            '''




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 600, 600) # wspolrzedne polozenia na ekranie (prawo/lewo, gora/dol), szerokosc, wysokosc
        #self.setWindowTitle("Pixel Grid")
        self.setWindowTitle("Zadanie domowe")
        self.grid = Grid(28,28,25)
        self.setCentralWidget(self.grid)
        self.show()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())





