#!/usr/bin/env python

from numpy.linalg import LinAlgError

from models.math import Scalar, Vector, Matrix
from utils.argparser import ModifiedArgumentParser
from utils.input import parse_matrix

if __name__ == '__main__':
    parser = ModifiedArgumentParser(prog='Lab1 for TRPO')
    subparsers = parser.add_subparsers(
        help='Операции над различными примитивами')

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
    matrix_operations.add_argument(
        '-s',
        '--sum',
        action='store_true',
        help='Поэлементное сложение',
        dest='mat_sum')
    matrix_operations.add_argument(
        '-m',
        '--mul',
        action='store_true',
        help='Поэлементное умножение',
        dest='mat_mul')
    matrix_operations.add_argument(
        '-mp',
        '--mat_mul',
        action='store_true',
        help='Матричное произведение',
        dest='mat_matmul')
    matrix_operations.add_argument(
        '-d',
        '--det',
        action='store_true',
        help='Определитель матрицы',
        dest='mat_det')
    matrix_operations.add_argument(
        '-r',
        '--reverse',
        action='store_true',
        help='Обратная матрицы',
        dest='mat_reverse')
    matrix_operations.add_argument(
        '-t',
        '--transp',
        action='store_true',
        help='Транспонирование матрицы',
        dest='mat_transp')

    matrix_operations.add_argument(
        '--trace',
        action='store_true',
        help='След матрицы',
        dest='mat_trace')
    matrix_operations.add_argument(
        '--mul_scal',
        action='store_true',
        help='Умножение матрицы на скаляр',
        dest='mat_on_scal')
    matrix_operations.add_argument(
        '--mul_vector',
        action='store_true',
        help='Умножение матрицы на вектор',
        dest='mat_on_vec')

    args = parser.parse_args()

    ########################
    #       Матрицы
    #######################

    # Поэлементное сложение
    if getattr(args, 'mat_sum', False):
        first_matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n'
        )
        second_matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n'
        )

        print(
            Matrix(
                parse_matrix(first_matrix)
            ) + Matrix(
                parse_matrix(second_matrix)
            )
        )

    # Поэлементное умножение
    elif getattr(args, 'mat_mul', False):
        first_matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n'
        )
        second_matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n'
        )

        print(
            Matrix(
                parse_matrix(first_matrix)
            ).mul_by_element(
                Matrix(
                    parse_matrix(second_matrix)
                )
            )
        )

    # Матричное произведение
    elif getattr(args, 'mat_matmul', False):
        first_matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n'
        )
        second_matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n'
        )

        print(
            Matrix(
                parse_matrix(first_matrix)
            ) * Matrix(
                parse_matrix(second_matrix)
            )
        )

    # Определитель матрицы
    elif getattr(args, 'mat_det', False):
        matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n'
        )

        try:
            print(
                Matrix(
                    parse_matrix(matrix)
                ).det
            )
        except LinAlgError:
            parser.exit_with_error('Матрица должна быть квадратной')

    # След матрицы
    elif getattr(args, 'mat_trace', False):
        matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n'
        )

        print(
            Matrix(
                parse_matrix(matrix)
            ).trace
        )

    # Обратная матрица
    elif getattr(args, 'mat_reverse', False):
        matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n'
        )

        try:
            print(
                Matrix(
                    parse_matrix(matrix)
                ).reverse
            )
        except LinAlgError:
            parser.exit_with_error('Матрица должна быть квадратной')

    # Транспонирование матрицы
    elif getattr(args, 'mat_transp', False):
        matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n'
        )

        print(
            Matrix(
                parse_matrix(matrix)
            ).T
        )

    # Умножение матрицы на скаляр
    elif getattr(args, 'mat_on_scal', False):
        scal = float(input('Введите скаляр:\n'))
        matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n'
        )

        print(
            Matrix(
                parse_matrix(matrix)
            ) * Scalar(scal)
        )

    # Умножение матрицы на вектор
    elif getattr(args, 'mat_on_vec', False):
        matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n')
        vector = input(
            'Введите первый вектор, элементы вводятся через запятую:\n')

        print(
            Matrix(
                parse_matrix(
                    matrix
                )
            ).mul_on_vector(
                Vector(
                    [float(i) for i in vector.split(',')]
                )
            )
        )

    ########################
    #       Вектора
    #######################

    # Умножение вектора на скаляр
    elif getattr(args, 'vec_on_scal', False):
        vec = input('Введите первый вектор, элементы вводятся через запятую:\n')
        scal = float(input('Введите скаляр:\n'))

        result = Vector(
            [float(i) for i in vec.split(',')]
        ) * Scalar(scal)

        print(result)

    # Поэлементное сложение векторов
    elif getattr(args, 'vec_sum', False):
        first_vec = input(
            'Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input(
            'Введите второй вектор, элементы вводятся через запятую:\n')

        print(
            Vector([float(i) for i in first_vec.split(',')]) +
            Vector([float(i) for i in second_vec.split(',')])
        )

    # Поэлементное умножение векторов
    elif getattr(args, 'vec_mul', False):
        first_vec = input(
            'Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input(
            'Введите второй вектор, элементы вводятся через запятую:\n')

        print(
            Vector([float(i) for i in first_vec.split(',')]) *
            Vector([float(i) for i in second_vec.split(',')])
        )

    # Скалярное произведение векторов
    elif getattr(args, 'vec_scal', False):
        first_vec = input(
            'Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input(
            'Введите второй вектор, элементы вводятся через запятую:\n')

        print(
            Vector(
                [float(i) for i in first_vec.split(',')]
            ).scalar_mul(
                Vector([float(i) for i in second_vec.split(',')])
            )
        )

    # Векторное произведение трехмерных векторов
    elif getattr(args, 'vec_vecmul', False):
        first_vec = input(
            'Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input(
            'Введите второй вектор, элементы вводятся через запятую:\n')

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
        vector = input(
            'Введите первый вектор, элементы вводятся через запятую:\n')

        print(
            Vector(
                [float(i) for i in vector.split(',')]
            ).vec_len
        )

    # Проверка сонаправленности векторов
    elif getattr(args, 'vec_cocheck', False):
        first_vec = input(
            'Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input(
            'Введите второй вектор, элементы вводятся через запятую:\n')

        print(
            Vector(
                [float(i) for i in first_vec.split(',')]
            ).is_collen(
                Vector([float(i) for i in second_vec.split(',')])
            )
        )

    # Проверка векторов на ортогональность
    elif getattr(args, 'vec_is_ortog', False):
        first_vec = input(
            'Введите первый вектор, элементы вводятся через запятую:\n')
        second_vec = input(
            'Введите второй вектор, элементы вводятся через запятую:\n')

        print(
            Vector(
                [float(i) for i in first_vec.split(',')]
            ).is_ortog(
                Vector([float(i) for i in second_vec.split(',')])
            )
        )

    # Умножение вектора на матрицу
    elif getattr(args, 'vec_on_matr', False):
        vector = input(
            'Введите первый вектор, элементы вводятся через запятую:\n')
        matrix = input(
            'Введите матрицу, столбцы разделяются через ;, элементы - ,:\n')

        print(
            Vector(
                [float(i) for i in vector.split(',')]
            ).mul_on_matrix(
                Matrix(
                    parse_matrix(
                        matrix
                    )
                )
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

    else:
        parser.print_help_and_exit()
