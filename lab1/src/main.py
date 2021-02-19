#!/usr/bin/env python

from argparse import ArgumentParser, Namespace
import numpy as np
from math import sin, cos, tan
from sys import stderr


class ModifiedArgumentParser(ArgumentParser):
    def exit_with_error(self, error_text):
        print(error_text, file=stderr)
        exit(1)

    def print_help_and_exit(self):
        self.print_help()
        exit(0)

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
        elif isinstance(other, Scalar):
            return Vector(self.__data * other.val)
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

    def scalar_mul(self, other) -> np.int64:
        '''
        Скалярное умножение векторов
        '''

        if not isinstance(other, Vector):
            raise TypeError('Второй аргумент не является вектором')
        else:
            return np.dot(self.__data, other.numpy_repr)

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

    def is_collen(self, second_vector) -> bool:
        if not isinstance(second_vector, Vector):
            raise TypeError('Второй аргумент не является вектором')
        else:
            if second_vector.len != self.len:
                raise ValueError('Длины векторов не равны')
            else:
                second_vector = second_vector.numpy_repr

            results = []

            for i, j in zip(self.__data, second_vector):
                results.append(i / j)

            if len(set(results)) == 1:
                return True

            return False

    def is_ortog(self, second_vector) -> bool:
        if self.scalar_mul(second_vector) == 0:
            return True

        return False


class Matrix(np.matrix):
    def __mul__(self, other):
        if isinstance(other, Scalar):
            return Matrix(
                self.A.dot(other.val)
            )
        else:
            return self.A * other

    @property
    def reverse(self):
        """
        Обратная матрица

        Если матрица не является квадратной,
        то возвращает исключение <<numpy.linalg.LinAlgError>>
        """

        return Matrix(np.linalg.inv(self.A))

    @property
    def trace(self) -> np.int64:
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
            return Matrix(np.multiply(self.A, matrix.A))

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
                return Matrix(
                    f.dot(
                        vec.numpy_repr
                    )
                )


class Scalar:
    def __init__(self, data) -> None:
        self.__data = data

    def __str__(self) -> str:
        return str(self.__data)

    def __add__(self, other):
        if isinstance(other, Scalar):
            return self.__data + other.val
        else:
            return self.__data + other

    def __mul__(self, other):
        if isinstance(other, Scalar):
            return self.__data * other.val
        else:
            return self.__data * other

    @property
    def val(self):
        return self.__data

    @property
    def reverse(self):
        '''
        Обратное
        '''

        return Scalar(-self.__data)

    @property
    def sin(self):
        return Scalar(
            sin(self.__data)
        )

    @property
    def cos(self):
        return Scalar(
            cos(self.__data)
        )

    @property
    def tan(self):
        return Scalar(
            tan(self.__data)
        )

    @property
    def ctg(self):
        return Scalar(
            1 / tan(self.__data)
        )

    def pow(self, other):
        '''
        Возведение в степень
        '''

        return Scalar(
            self.__data ** other
        )

    def sqrt(self, other):
        '''
        Извлечение корня
        '''

        return Scalar(
            self.__data ** (1 / other)
        )


if __name__ == '__main__':
    parser = ModifiedArgumentParser(prog='Lab1 for TRPO')
    subparsers = parser.add_subparsers(help='sub-command help')

    scalar_operations = subparsers.add_parser(
        'scalar', help='Операции над скалярами')
    scalar_operations.add_argument(
        '-s',
        '--sum',
        nargs=2,
        type=float,
        help='Сумма',
        dest='scal_sum')
    scalar_operations.add_argument(
        '-r',
        '--reverse',
        type=float,
        help='Обратное',
        dest='scal_reverse')
    scalar_operations.add_argument(
        '-m',
        '--mul',
        nargs=2,
        type=float,
        help='Умножение',
        dest='scal_mul')
    scalar_operations.add_argument(
        '--pow',
        nargs=2,
        type=float,
        help='Возведение в степень',
        dest='scal_pow')
    scalar_operations.add_argument(
        '--sqrt',
        nargs=2,
        type=float,
        help='Извлечение корня',
        dest='scal_sqrt')
    scalar_operations.add_argument(
        '--sin',
        type=float,
        help='Синус',
        dest='scal_sin')
    scalar_operations.add_argument(
        '--cos',
        type=float,
        help='Косинус',
        dest='scal_cos')
    scalar_operations.add_argument(
        '--tan',
        type=float,
        help='Тангенс',
        dest='scal_tan')
    scalar_operations.add_argument(
        '--ctg',
        type=float,
        help='Котангенс',
        dest='scal_ctg')

    vector_operations = subparsers.add_parser(
        'vector', help='Операции над векторами')
    vector_operations.add_argument(
        '-s',
        '--sum',
        action='store_true',
        help='Поэлементное сложение',
        dest='vec_sum')
    vector_operations.add_argument(
        '-m',
        '--mul',
        action='store_true',
        help='Поэлементное умножение',
        dest='vec_mul')
    vector_operations.add_argument(
        '-sp',
        '--scalar',
        action='store_true',
        help='Скалярное произведение',
        dest='vec_scal')
    vector_operations.add_argument(
        '-vm',
        '--vec_mul',
        action='store_true',
        help='Векторное произведение',
        dest='vec_vecmul')
    vector_operations.add_argument(
        '-l',
        '--len',
        action='store_true',
        help='Длина вектора',
        dest='vec_len')
    vector_operations.add_argument(
        '--codirect',
        action='store_true',
        help='Проверка сонаправленности векторов',
        dest='vec_cocheck')
    vector_operations.add_argument(
        '--is_ortog',
        action='store_true',
        help='Проверка векторов на ортогональность',
        dest='vec_is_ortog')
    vector_operations.add_argument(
        '--mul_matr',
        action='store_true',
        help='Умножение вектора на матрицу',
        dest='vec_on_matr')
    vector_operations.add_argument(
        '--mul_scal',
        action='store_true',
        help='Умножение вектора на скаляр',
        dest='vec_on_scal')

    matrix_operations = subparsers.add_parser(
        'matrix', help='Операции над матрицами')
    
    args = parser.parse_args()
            
    ########################
    #       Вектора
    #######################
    
    # Умножение вектора на скаляр
    if getattr(args, 'vec_on_scal', False):
        vec = input('Введите первый вектор, элементы вводятся через запятую:\n')
        scal = float(input('Введите скаляр:\n'))

        result = Vector(
            [float(i) for i in vec.split(',')]
        ) * Scalar(scal)

        print(result)

    # Поэлементное сложение векторов
    elif getattr(args, 'vec_sum', False):
        first_vec = input('Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input('Введите второй вектор, элементы вводятся через запятую:\n')
        
        print(
            Vector([float(i) for i in first_vec.split(',')]) +
            Vector([float(i) for i in second_vec.split(',')])
        )

    # Поэлементное умножение векторов
    elif getattr(args, 'vec_mul', False):
        first_vec = input('Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input('Введите второй вектор, элементы вводятся через запятую:\n')

        print(
            Vector([float(i) for i in first_vec.split(',')]) *
            Vector([float(i) for i in second_vec.split(',')])
        )

    # Скалярное произведение векторов
    elif getattr(args, 'vec_scal', False):
        first_vec = input('Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input('Введите второй вектор, элементы вводятся через запятую:\n')

        print(
            Vector(
                [float(i) for i in first_vec.split(',')]
            ).scalar_mul(
                Vector([float(i) for i in second_vec.split(',')])
            )
        )

    # Векторное произведение трехмерных векторов
    elif getattr(args, 'vec_vecmul', False):
        first_vec = input('Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input('Введите второй вектор, элементы вводятся через запятую:\n')

        try:
            print(
                Vector(
                    [float(i) for i in first_vec.split(',')]
                ).vec_mul(
                    Vector([float(i) for i in second_vec.split(',')])
                )
            )
        except ValueError:
            parser.exit_with_error('Оба вектора должны быть трехмерными')

    # Длина вектора
    elif getattr(args, 'vec_len', False):
        vector = input('Введите первый вектор, элементы вводятся через запятую:\n')

        print(
            Vector(
                [float(i) for i in vector.split(',')]
            ).vec_len
        )

    # Проверка сонаправленности векторов
    elif getattr(args, 'vec_cocheck', False):
        first_vec = input('Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input('Введите второй вектор, элементы вводятся через запятую:\n')
        
        print(
            Vector(
                [float(i) for i in first_vec.split(',')]
            ).is_collen(
                Vector([float(i) for i in second_vec.split(',')])
            )
        )

    # Проверка векторов на ортогональность
    elif getattr(args, 'vec_is_ortog', False):
        first_vec = input('Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input('Введите второй вектор, элементы вводятся через запятую:\n')

        print(
            Vector(
                [float(i) for i in first_vec.split(',')]
            ).is_ortog(
                Vector([float(i) for i in second_vec.split(',')])
            )
        )

    # Умножение вектора на матрицу
    elif getattr(args, 'vec_on_matr', False):
        first_vec = input('Введите первый вектор, элементы вводятся через запятую:\n')
        input_matrix = input('Введите матрицу, столбцы разделяются через ;, элементы - ,:\n')

        matrix = []
        tmp = []

        for i in input_matrix:
            if i != ',' and i != ';':
                tmp.append(float(i))

            if i == ';':
                matrix.append(tmp)
                tmp = []

        if len(tmp) != 0:
            matrix.append(tmp)

        print(
            Vector(
                [float(i) for i in first_vec.split(',')]
            ).mul_on_matrix(
                Matrix(matrix)
            )
        )

    ########################
    #       Скаляры
    #######################

    # Сумма скаляров
    elif getattr(args, 'scal_sum', False):
        first_sc, second_sc = args.scal_sum

        print(Scalar(first_sc) + Scalar(second_sc))

    # Инверсия скаляра
    elif getattr(args, 'scal_reverse', False):
        print(Scalar(args.scal_reverse).reverse)

    # Произведение скаляров
    elif getattr(args, 'scal_mul', False):
        first_sc, second_sc = args.scal_mul

        print(Scalar(first_sc) * Scalar(second_sc))

    # Возведение в степень скаляра
    elif getattr(args, 'scal_pow', False):
        first_sc, pow_val = args.scal_pow

        print(Scalar(first_sc).pow(pow_val))

    # Вычисление корня из скаляра
    elif getattr(args, 'scal_sqrt', False):
        first_sc, sqrt_val = args.scal_sqrt

        print(Scalar(first_sc).sqrt(sqrt_val))

    # Sin скаляра
    elif getattr(args, 'scal_sin', False):
        print(Scalar(args.scal_sin).sin)

    # Cos скаляра
    elif getattr(args, 'scal_cos', False):
        print(Scalar(args.scal_cos).cos)

    # Tan скаляра
    elif getattr(args, 'scal_tan', False):
        print(Scalar(args.scal_tan).tan)

    # Ctg скаляра
    elif getattr(args, 'scal_ctg', False):
        print(Scalar(args.scal_ctg).ctg)
    
    #f = Matrix([[1, 2, 3], [4, 5, 6]])
    #s = Matrix([[1, 2], [4, 5], [3, 6]])
    #obr_matr = Matrix([[1, 2], [3, 4]])
    #vec = Vector([1, 2, 3])
    #sc = Scalar(2)
    #sc_f = Scalar(2.0)

    # matrix on scalar
    # print(f * sc)

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
