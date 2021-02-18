#!/usr/bin/env python

import numpy as np


class Vector:
    def __init__(self, data: list) -> None:
        self.__data = np.array(data)

    def __str__(self) -> str:
        return str(self.__data)

    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            return Vector(self.__data * other)
        elif type(other) == Vector:
            if len(other.numpy_repr) != self.len:
                raise ValueError('Не совпадает длина векторов')
            else:
                return Vector(self.__data * other.numpy_repr)

    def __add__(self, other):
        if type(other) != Vector:
            raise TypeError('Второй аргумент не является вектором')
        else:
            if len(other.numpy_repr) != self.len:
                raise ValueError('Не совпадает длина векторов')
            else:
                return Vector(self.__data + other.numpy_repr)

    @property
    def numpy_repr(self) -> np.array:
        return self.__data

    @property
    def len(self) -> int:
        return len(self.__data)

    @property
    def vec_len(self) -> float:
        return np.linalg.norm(self.__data)
    
    def mul_on_matrix(self, matrix):
        if type(matrix) != Matrix:
            raise TypeError('Второй аргумент не является матрицей')
        else:
            if self.len != len(matrix.A[0]):
                raise ValueError('Не совпадает длина')
            else:
                return matrix.dot(
                    self.__data
                )

    def scalar_mul(self, other):
        if type(other) != Vector:
            raise TypeError('Второй аргумент не является вектором')
        else:
            return Vector(np.dot(self.__data, other.numpy_repr))

    def vec_mul(self, other):
        if type(other) != Vector:
            raise TypeError('Второй аргумент не является вектором')
        else:
            if self.len < 3 or other.len < 3:
                raise ValueError('Вектор не является трехмерным')
            else:
                return Vector(np.cross(self.__data, other.numpy_repr))

class Matrix(np.matrix):
    def mul_on_vector(self, vector):
        if type(vector) != Vector:
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

    # поэлементное произведение
    # print(np.multiply(f, f))

    # mul matrix on vector
    # когда число столбцов матрицы равно числу строк вектора
    # print(f.mul_on_vector(vec))

    # mul
    # число столбцов матрицы А равно числу строк матрицы В
    # print(f * s)

    # след
    # print(np.trace(f))

    # обратная матрица
    # должна быть квадратной
    # try:
    #     print(np.linalg.inv(obr_matr))
    # except numpy.linalg.LinAlgError:
    #     print('Нельзя сделать обратную')

    # транспонирование
    # print(f.T)
