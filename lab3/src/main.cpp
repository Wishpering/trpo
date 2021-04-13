#include <iostream>
#include <cmath>
#include <ctime>
#include <omp.h>
using namespace std;

int main() {
  long int R, X, Y; // R - радиус окружности
  
  // X, Y - координаты сгенерированной точки
  double start_time, end_time, //Время начала и конца выполнения программы
    Pi, chetcik; //полученное число пи

  // iteraci - количество брошенных точек
  // chetcik - количество попавших точек

  long int iteraci, // iteraci - количество брошенных точек
    i; // i - счетчик цикла

  //Запоминаем время начала работы программы
  // start_time = clock(); //iteraci - количество брошенных точек

  start_time = omp_get_wtime();
  
  //Объявляем переменные
  R = 3276;
  iteraci = pow(10, 9);
  chetcik = 0;

  //Запускаем циклы параллельно
  //основной цикл
  omp_set_num_threads(4);
  omp_set_dynamic(true);
  
  #pragma omp parallel for private(i, X, Y) reduction(+ : chetcik)
  for (i = 0; i < iteraci; i++) {
    X = rand() % R;
    Y = rand() % R;

    if ((X * X + Y * Y) <= R * R) chetcik++;
  }

  //Вычисляем число пи
  Pi = 4 * (chetcik / iteraci);

  //Узнаем время окончания работы программы

  // end_time = clock();
  end_time = omp_get_wtime();

  //Выводим результат и время, за которое мы получили результат
  cout << "Число пи примерно равно: " << Pi << endl;
  cout << "Время выполнения программы: " << (end_time - start_time) << endl;
}
