#include <iostream>
#include <cmath>
#include <random>

int main() {
  const size_t iteration_count = pow(10, 9),
    R = 4000;
  
  static size_t iteration_num = 0;
  double chetcik;
  unsigned long int register X, Y;

  std::random_device rd;
  std::mt19937_64 generator(rd());
  std::uniform_real_distribution<> dist(0, R);
    
  for (iteration_num = 0; iteration_num < iteration_count; iteration_num++) {
    X = dist(generator);
    Y = dist(generator);

    if ((X * X + Y * Y) <= R * R) chetcik++;
  }

  std::cout << "Примерно равно: " << 4 * (chetcik / iteration_count) << std::endl;
}
