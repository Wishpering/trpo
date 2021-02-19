# Описание

Лабораторная работа №1 по предмету "Практикум по технологии разработки программного обеспечения"

## Требования
Python 3.7+

## Запуск

```bash
pip install -r requirements.txt
python main.py
```

## Структура

```
lab1
│
└───latex  
│   │   lab1.tex
│   └───include
│       │   *.svg
│
└───src
│   │   main.py
│   │   requirements.txt
│   └───models
│   │   │   __init__.py
│   │ 	│   math.py	 - описание математических примитивов
│   └───utils
│       │   __init__.py
│       │   argparser.py - парсер входных аргументов
│       │   input.py     - обработка входных данных
```