#!/usr/bin/env bash

# Create build directory.
mkdir -p build
# cd into ./build
cd build
# Cmake
echo "Cmake: "
cmake ..
echo 
echo "Make: "
make
echo
echo "Execute: "
# Run my_app
./my_app
echo
