CC=g++
DEBUG=-O0 -g --std=c++11
RELEASE=-Ofast --std=c++11
OPT = $(DEBUG)
COPT= -c $(OPT) 
LOPT= $(OPT)

OBJ = mat_matrix.o

all: mat_matrix.o
	$(CC) $(LOPT) -o mat_matrix mat_matrix.o

clean:
	rm -f *.o *.gch

mat_matrix.o: mat_matrix.h mat_matrix.cpp
	$(CC) $(COPT) mat_matrix.cpp

