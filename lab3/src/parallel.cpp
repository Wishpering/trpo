#include <iostream>
#include <cmath>
#include <omp.h>
#include <random>

int main() {
  const size_t iteration_count = pow(10, 9),
    R = 4000;
  
  size_t iteration_num = 0;
  uint64_t register X, Y;
  double chetcik;
  
  std::random_device rd;
  std::mt19937_64 generator(rd());
  std::uniform_real_distribution<> dist(0, R);
    
  omp_set_num_threads(16);
  omp_set_dynamic(true);
  
  #pragma omp parallel for private(iteration_num, X, Y) reduction(+ : chetcik)
  for (iteration_num = 0; iteration_num < iteration_count; iteration_num++) {
    X = dist(generator);
    Y = dist(generator);

    if ((X * X + Y * Y) <= R * R) chetcik++;
  }

  std::cout << "Примерно равно: " << 4 * (chetcik / iteration_count) << std::endl;
}
