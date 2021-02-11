#!/usr/bin/env python

from numpy import matrix as np_matrix
from random import SystemRandom
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QMessageBox, QApplication,
    QMainWindow)
from PyQt5.QtCore import Qt
from sys import argv


class Matrix:
    '''
        x
      ------
    y |    |
      ------
    '''

    def __init__(self, x=0, y=0):
        self.__x = x
        self.__y = y

        self.__matrix = []

    def fill(self):
        generator = SystemRandom()

        for i in range(self.__x):
            tmp = []

            for j in range(self.__y):
                tmp.append(generator.randint(0, 100))

            self.__matrix.append(tmp)

    @property
    def numpy_matrix(self):
        return np_matrix(self.__matrix)

    @property
    def x_size(self):
        return self.__x

    @property
    def y_size(self):
        return self.__y

    @x_size.setter
    def x_size(self, new_x):
        self.__x = new_x

    @y_size.setter
    def y_size(self, new_y):
        self.__y = new_y


class ErrorWindow(QMessageBox):
    '''Представляет собой окно с ошибкой'''

    def __init__(self, error_text):
        '''error_text - текст ошибки'''

        super().__init__()

        self.setIcon(QMessageBox.Critical)
        self.setText('Error')
        self.setInformativeText(error_text)
        self.setWindowTitle('Error')
        self.exec()


class MainWindow(QMainWindow):
    '''Представляет собой главное окно программы'''

    def __init__(self):
        super().__init__()
        uic.loadUi('./interfaces/main_window.ui', self)

        self.first_matrix, self.second_matrix = Matrix(), Matrix()
        self.start_button.clicked.connect(self.start_button_handler)

    def start_button_handler(self):
        try:
            self.first_matrix.x_size, self.first_matrix.y_size = int(
                self.N_input.text()), int(self.K_input.text())
            self.second_matrix.x_size, self.second_matrix.y_size = int(
                self.K_input.text()), int(self.M_input.text())
        except ValueError:
            ErrorWindow('Введеные значения не являются числами')

        self.first_matrix.fill()
        self.second_matrix.fill()

        print(self.first_matrix.numpy_matrix)
        print(self.second_matrix.numpy_matrix)
        
        print(
            self.first_matrix.numpy_matrix.dot(
                self.second_matrix.numpy_matrix)
        )


if __name__ == '__main__':
    app = QApplication(argv)
    mw = MainWindow()
    mw.show()
    exit(app.exec())
