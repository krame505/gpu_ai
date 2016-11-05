################################################################################
#
# Build script for project
#
################################################################################

# Using a newer version of CUDA
CUDA_INSTALL_PATH := /usr/local/cuda-6.5

EXECUTABLE	:= run_ai
# Cuda source files (compiled with cudacc)
CUFILES		:= state.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= mcts.cpp

include ../../common/common.mk

# Enable C++11 for compiling .cpp files
# NVCC doesn't support gcc5 yet, but we can still use gcc5 for general c++
# This hacky stuff needs to go after the include to override some variables that
# I'm not sure were meant to be overridden... but it seems to work
CXXFLAGS        += -std=gnu++11
NVCCFLAGS       += --compiler-bindir /usr/bin/g++
CXX              = /opt/rh/devtoolset-4/root/usr/bin/g++
CC               = /opt/rh/devtoolset-4/root/usr/bin/gcc