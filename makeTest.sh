#!/bin/bash

src=$( cat <<EOL
mat_matrix.cpp
test_mat_matrix.cpp
EOL
)

g++ -o test_mat_matrix $src --std=c++11
