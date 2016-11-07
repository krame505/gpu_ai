################################################################################
#
# Build script for project
#
################################################################################

# Using a newer version of CUDA than the default
CUDA_INSTALL_PATH := /usr/local/cuda-8.0

EXECUTABLE	:= run_ai
# Cuda source files (compiled with cudacc)
CUFILES		:= state.cu playout.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= mcts.cpp player.cpp driver.cpp
# Header files / anything that should trigger a full rebuild
C_DEPS          := mcts.hpp playout.hpp state.hpp player.cpp

include ../../common/common.mk

# Enable C++11 for compiling .cpp files
CXXFLAGS        += -std=gnu++11