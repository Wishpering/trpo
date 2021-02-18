#!/usr/bin/env python

import numpy as np


class Vector:
    def __init__(self, data: list) -> None:
        self.__data = np.array(data)

    def __str__(self) -> str:
        return str(self.__data)

    def __mul__(self, other):
        '''
        Перегрузка оператора умножения
        '''

        if isinstance(other, int) or isinstance(other, float):
            return Vector(self.__data * other)
        elif isinstance(other, Vector):
            if len(other.numpy_repr) != self.len:
                raise ValueError('Не совпадает длина векторов')
            else:
                return Vector(self.__data * other.numpy_repr)

    def __add__(self, other):
        '''
        Перегрузка оператора сложения
        '''

        if not isinstance(other, Vector):
            raise TypeError('Второй аргумент не является вектором')
        else:
            if len(other.numpy_repr) != self.len:
                raise ValueError('Не совпадает длина векторов')
            else:
                return Vector(self.__data + other.numpy_repr)

    @property
    def numpy_repr(self) -> np.array:
        '''
        Представление вектора в качестве <<NumPy.array>>
        '''

        return self.__data

    @property
    def len(self) -> int:
        '''
        Длина в представлении Python
        '''

        return len(self.__data)

    @property
    def vec_len(self) -> float:
        '''
        Длина вектора
        '''

        return np.linalg.norm(self.__data)

    def mul_on_matrix(self, matrix):
        '''
        Умножение вектора на матрицу
        '''

        if not isinstance(matrix, Matrix):
            raise TypeError('Второй аргумент не является матрицей')
        else:
            if self.len != len(matrix.A[0]):
                raise ValueError('Не совпадает длина')
            else:
                return Vector(
                    matrix.dot(
                        self.__data
                    )
                )

    def scalar_mul(self, other):
        '''
        Скалярное умножение векторов
        '''

        if not isinstance(other, Vector):
            raise TypeError('Второй аргумент не является вектором')
        else:
            return Vector(np.dot(self.__data, other.numpy_repr))

    def vec_mul(self, other):
        '''
        Векторное произведение трехмерный векторов
        '''

        if not isinstance(other, Vector):
            raise TypeError('Второй аргумент не является вектором')
        else:
            if self.len < 3 or other.len < 3:
                raise ValueError('Вектор не является трехмерным')
            else:
                return Vector(np.cross(self.__data, other.numpy_repr))


class Matrix(np.matrix):
    @property
    def reverse(self):
        """
        Обратная матрица

        Если матрица не является квадратной,
        то возвращает исключение <<numpy.linalg.LinAlgError>>
        """

        return np.linalg.inv(self.A)

    @property
    def trace(self):
        '''
        След матрицы
        '''

        return np.trace(self.A)

    def mul_by_element(self, matrix):
        '''
        Поэлементное произведение матрицы на матрицу
        '''

        if not isinstance(matrix, Matrix):
            raise TypeError('Вторая аргумент не является NumPy матрицой')
        else:
            return np.multiply(self.A, matrix.A)

    def mul_on_vector(self, vector):
        '''
        Умножение матрицы на вектор
        '''

        if not isinstance(vector, Vector):
            raise TypeError('Второй аргумент не является вектором')
        else:
            if vector.len != len(self.A[0]):
                raise ValueError('Не совпадает длина')
            else:
                return f.dot(
                    vec.numpy_repr
                )


if __name__ == '__main__':
    f = Matrix([[1, 2, 3], [4, 5, 6]])
    s = Matrix([[1, 2], [4, 5], [3, 6]])
    obr_matr = Matrix([[1, 2], [3, 4]])
    vec = Vector([1, 2, 3])

    # matrix on scalar
    # print(f * 2)

    # sum matrix
    # матрицы должны быть одинаковы по размеру
    # print(f + f)

    # поэлементное умножение
    # print(f.mul_by_element(f))

    # mul matrix on vector
    # когда число столбцов матрицы равно числу строк вектора
    # print(f.mul_on_vector(vec))

    # mul
    # число столбцов матрицы А равно числу строк матрицы В
    # print(f * s)

    # след
    # print(f.trace)

    # обратная матрица
    # должна быть квадратной
    # try:
    #     print(obr_matr.reverse)
    # except numpy.linalg.LinAlgError:
    #     print('Нельзя сделать обратную')

    # транспонирование
    # print(f.T)
