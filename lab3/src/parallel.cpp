#include <iostream>
#include <cmath>
#include <omp.h>
#include <random>

float randFloat(double low, double high) {
    thread_local static std::random_device rd;
    thread_local static std::mt19937 rng(rd());
    thread_local std::uniform_real_distribution<double> urd;
    return urd(rng, decltype(urd)::param_type{low, high});
}

int main() {
  const size_t iteration_count = pow(10, 9),
    R = 4000;
  
  size_t iteration_num = 0;
  uint64_t register X, Y;
  double chetcik;
      
  omp_set_num_threads(16);
  omp_set_dynamic(true);
  
  #pragma omp parallel for private(iteration_num, X, Y) reduction(+ : chetcik)
  for (iteration_num = 0; iteration_num < iteration_count; iteration_num++) {    
    X = randFloat(0, R);
    Y = randFloat(0, R);
      
    if ((X * X + Y * Y) <= R * R) chetcik++;
  }

  std::cout << "Примерно равно: " << 4 * (chetcik / iteration_count) << std::endl;
}
