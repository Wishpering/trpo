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
        help='Обратное',
        dest='scal_sin')
    scalar_operations.add_argument(
        '--cos',
        type=float,
        help='Обратное',
        dest='scal_cos')
    scalar_operations.add_argument(
        '--tan',
        type=float,
        help='Обратное',
        dest='scal_tan')
    scalar_operations.add_argument(
        '--ctg',
        type=float,
        help='Обратное',
        dest='scal_ctg')

    vector_operations = subparsers.add_parser(
        'vector', help='Операции над векторами')
    vector_operations.add_argument(
        '-s',
        '--sum',
        nargs=2,
        type=list,
        help='Поэлементное сложение',
        dest='vec_sum')
    vector_operations.add_argument(
        '-m',
        '--mul',
        nargs=2,
        type=list,
        help='Поэлементное умножение',
        dest='vec_mul')
    vector_operations.add_argument(
        '-sp',
        '--scalar',
        nargs=2,
        type=list,
        help='Скалярное произведение',
        dest='vec_scal')
    vector_operations.add_argument(
        '-vm',
        '--vec_mul',
        nargs=2,
        type=list,
        help='Векторное произведение',
        dest='vec_vecmul')
    vector_operations.add_argument(
        '-l',
        '--len',
        type=list,
        help='Длина вектора',
        dest='vec_len')
    vector_operations.add_argument(
        '--codirect',
        nargs=2,
        type=list,
        help='Проверка сонаправленности векторов',
        dest='vec_cocheck')
    vector_operations.add_argument(
        '--is_ortog',
        nargs=2,
        type=list,
        help='Проверка векторов на ортогональность',
        dest='vec_is_ortog')
    vector_operations.add_argument(
        '--mul_matr',
        nargs=2,
        type=list,
        help='Умножение вектора на матрицу',
        dest='vec_on_matr')
    vector_operations.add_argument(
        '--mul_scal',
        nargs=2,
        type=list,
        help='Умножение вектора на скаляр',
        dest='vec_on_scal')

    matrix_operations = subparsers.add_parser(
        'matrix', help='Операции над матрицами')
    
    args = parser.parse_args()

    dict_args = vars(args)
    if len(dict_args) == 0:
        parser.print_help_and_exit()
    else:
        any_set = False
        
        for i, j in dict_args.items():
            if j != None:
                any_set = True

        if not any_set:
            parser.print_help_and_exit()
            
    ########################
    #       Вектора
    #######################
    
    # Умножение вектора на скаляр
    if args.vec_on_scal:
        vec, scal = args.vec_on_scal

        print(
            Vector([float(i) for i in vec if i != ',']) *
            Scalar(float(scal[0]))
        )

    # Поэлементное сложение векторов
    elif args.vec_sum:
        first_vec, second_vec = args.vec_sum

        print(
            Vector([float(i) for i in first_vec if i != ',']) +
            Vector([float(i) for i in second_vec if i != ','])
        )

    # Поэлементное умножение векторов
    elif args.vec_mul:
        first_vec, second_vec = args.vec_mul

        print(
            Vector([float(i) for i in first_vec if i != ',']) *
            Vector([float(i) for i in second_vec if i != ','])
        )

    # Скалярное произведение векторов
    elif args.vec_scal:
        first_vec, second_vec = args.vec_scal

        print(
            Vector(
                [float(i) for i in first_vec if i != ',']
            ).scalar_mul(
                Vector([float(i) for i in second_vec if i != ','])
            )
        )

    # Векторное произведение трехмерных векторов
    elif args.vec_vecmul:
        first_vec, second_vec = args.vec_vecmul

        try:
            print(
                Vector(
                    [float(i) for i in first_vec if i != ',']
                ).vec_mul(
                    Vector([float(i) for i in second_vec if i != ','])
                )
            )
        except ValueError:
            parser.exit_with_error('Вектора должна быть трехмерными')

    # Длина вектора
    elif args.vec_len:
        first_vec = args.vec_len

        print(
            Vector(
                [float(i) for i in first_vec if i != ',']
            ).vec_len
        )

    # Проверка сонаправленности векторов
    elif args.vec_cocheck:
        first_vec, second_vec = args.vec_cocheck

        print(
            Vector(
                [float(i) for i in first_vec if i != ',']
            ).is_collen(
                Vector([float(i) for i in second_vec if i != ','])
            )
        )

    # Проверка векторов на ортогональность
    elif args.vec_is_ortog:
        first_vec, second_vec = args.vec_is_ortog

        print(
            Vector(
                [float(i) for i in first_vec if i != ',']
            ).is_ortog(
                Vector([float(i) for i in second_vec if i != ','])
            )
        )

    # Умножение вектора на матрицу
    elif args.vec_on_matr:
        first_vec, input_matrix = args.vec_on_matr

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
                [float(i) for i in first_vec if i != ',']
            ).mul_on_matrix(
                Matrix(matrix)
            )
        )
    else:
        parser.print_help_and_exit()

    ########################
    #       Скаляры
    #######################

    # Сумма скаляров
    if args.scal_sum:
        first_sc, second_sc = args.scal_sum

        print(Scalar(first_sc) + Scalar(second_sc))

    # Инверсия скаляра
    elif args.scal_reverse:
        print(Scalar(args.scal_reverse).reverse)

    # Произведение скаляров
    elif args.scal_mul:
        first_sc, second_sc = args.scal_mul

        print(Scalar(first_sc) * Scalar(second_sc))

    # Возведение в степень скаляра
    elif args.scal_pow:
        first_sc, pow_val = args.scal_pow

        print(Scalar(first_sc).pow(pow_val))

    # Вычисление корня из скаляра
    elif args.scal_sqrt:
        first_sc, sqrt_val = args.scal_sqrt

        print(Scalar(first_sc).sqrt(sqrt_val))

    # Sin скаляра
    elif args.scal_sin:
        print(Scalar(args.scal_sin).sin)

    # Cos скаляра
    elif args.scal_cos:
        print(Scalar(args.scal_cos).cos)

    # Tan скаляра
    elif args.scal_tan:
        print(Scalar(args.scal_tan).tan)

    # Ctg скаляра
    elif args.scal_ctg:
        print(Scalar(args.scal_ctg).ctg)
    else:
        parser.print_help_and_exit()
    
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
