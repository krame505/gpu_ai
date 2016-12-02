################################################################################
#
# Build script for project
#
################################################################################

# Using a newer version of CUDA than the default
CUDA_INSTALL_PATH := /usr/local/cuda-8.0

EXECUTABLE	:= run_ai
# Cuda source files (compiled with cudacc)
CUFILES		:= state.cu playout.cu genMovesTest.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= state.cpp playout.cpp mcts.cpp player.cpp driver.cpp
# Header files / anything that should trigger a full rebuild
C_DEPS          := colors.h mcts.hpp playout.hpp state.hpp player.hpp
CU_DEPS         := playout.hpp state.hpp genMovesTest.hpp

#SMVERSIONFLAGS  := -arch=sm_20

include ../../common/common.mk

# Some overrides...
# Enable C++11 for compiling .cpp files
CXXFLAGS        += -std=gnu++11

NVCCFLAGS       += -dc
LINK            := $(NVCC)

ifeq ($(nounicode),1)
  CXXFLAGS      += -DNOUNICODE
endif