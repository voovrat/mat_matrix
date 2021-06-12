matlab-like matrix in C++

read mat_matrix.cpp for examples

fft_matrix provides the filtering for matrices.
It uses either fftw, or yaft (https://github.com/voovrat/yaft).
Note, that fftw implementation is not thread-safe, cause it uses the global fftw plans 
(yaft implementation is a pure-funtion implementation, so there should not be any problems with multithreading ). 


