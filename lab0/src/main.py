#!/usr/bin/env python

from numpy import matrix, savetxt
from random import SystemRandom
from os import getcwd, mkdir
from os.path import exists
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

    def __init__(self, x=0, y=0, **kwargs):
        '''
        x,y [int] - размер матрицы
        **kwargs:
                generator - Random generator
        '''

        self.__x = x
        self.__y = y

        self.__matrix = []

        self.generator = kwargs.get('generator') or SystemRandom()

    def fill(self):
        '''
        Заполнение матрицы случайными числами
        '''

        for i in range(self.__x):
            tmp = []

            for j in range(self.__y):
                tmp.append(self.generator.randint(0, 100))

            self.__matrix.append(tmp)

    @property
    def numpy_matrix(self) -> matrix:
        return matrix(self.__matrix)

    @property
    def x_size(self) -> int:
        return self.__x

    @property
    def y_size(self) -> int:
        return self.__y

    @x_size.setter
    def x_size(self, new_x: int):
        self.__x = new_x

    @y_size.setter
    def y_size(self, new_y: int):
        self.__y = new_y


class MessageWindow(QMessageBox):
    '''Представляет собой окно с текстом'''

    def __init__(self, text):
        '''text - текст для отображения'''

        super().__init__()

        self.setIcon(QMessageBox.Information)
        self.setText('Information')
        self.setInformativeText(text)
        self.setWindowTitle('Information')
        self.exec()


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

        generator = SystemRandom()

        self.first_matrix, self.second_matrix = Matrix(
            generator=generator
        ), Matrix(
            generator=generator
        )

        self.start_button.clicked.connect(self.start_button_handler)

    def start_button_handler(self):
        '''
        Обработчик нажатия кнопки <<start>>
        '''

        try:
            self.first_matrix.x_size, self.first_matrix.y_size = int(
                self.N_input.text()), int(self.K_input.text())
            self.second_matrix.x_size, self.second_matrix.y_size = int(
                self.K_input.text()), int(self.M_input.text())
        except ValueError:
            ErrorWindow('Введеные значения не являются числами')
            exit(0)

        self.first_matrix.fill()
        self.second_matrix.fill()

        self.first_matrix = self.first_matrix.numpy_matrix
        self.second_matrix = self.second_matrix.numpy_matrix

        result_matrix = self.first_matrix.dot(
            self.second_matrix
        )

        path = getcwd()
        if not exists(f'{path}/files'):
            mkdir(f'{path}/files')

        savetxt(
            f'{path}/files/first_matrix.csv',
            self.first_matrix,
            delimiter=',',
            fmt='%d')
        savetxt(
            f'{path}/files/second_matrix.csv',
            self.second_matrix,
            delimiter=',',
            fmt='%d')
        savetxt(
            f'{path}/files/result_matrix.csv',
            result_matrix,
            delimiter=',',
            fmt='%d')

        MessageWindow('Дело сделано')
        exit(0)


if __name__ == '__main__':
    app = QApplication(argv)
    mw = MainWindow()
    mw.show()
    exit(app.exec())
