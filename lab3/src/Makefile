CC = g++
CFLAGS = -O3 -march=native

all: serial parallel

bench: all
	@echo "Benching serial code..."
	@time ./lab3_serial;
	@echo "Benching parallel code..."
	@time ./lab3_parallel;

serial:
	$(CC) $(CFLAGS) serial.cpp -o lab3_serial

parallel:
	$(CC) $(CFLAGS) -fopenmp parallel.cpp -o lab3_parallel

clean:
	rm -vf lab3_*
