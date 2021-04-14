#include <iostream>
#include <cmath>
#include <random>
#include <omp.h>

double rand_point(double low, double high) {
    thread_local static std::random_device rd;
    thread_local static std::mt19937 rng(rd());
    thread_local std::uniform_real_distribution<double> urd;
    return urd(rng, decltype(urd)::param_type{low, high});
}

int main() {
  const size_t iteration_count = pow(10, 9),
    R = 4000;

  uint16_t register X, Y;
  
  static size_t iteration_num = 0,
    omp_iter_num;
  
  omp_set_num_threads(16);
  omp_set_dynamic(true);
  
  #pragma omp parallel for private(iteration_num, X, Y) reduction(+ : omp_iter_num)
  for (iteration_num = 0; iteration_num < iteration_count; iteration_num++) {    
    X = rand_point(0, R);
    Y = rand_point(0, R);
      
    if ((X * X + Y * Y) <= R * R) omp_iter_num++;
  }

  std::cout << "Примерно равно: " << 4 * (omp_iter_num / iteration_count) << std::endl;
}
