################################################################################
#
# Build script for project
#
################################################################################

# Using a newer version of CUDA than the default
CUDA_INSTALL_PATH := /usr/local/cuda-8.0

EXECUTABLE  := run_ai

# CUDA source files (compiled with cudacc)
CUFILES	    := state.cu playout.cu genMovesTest.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES	    := state.cpp playout.cpp mcts.cpp player.cpp driver.cpp
# Header files included by any of CUFILES
CUHEADERS   := playout.hpp state.hpp genMovesTest.hpp
# Header files included by any of CCFILES
CCHEADERS   := colors.h mcts.hpp playout.hpp state.hpp player.hpp

SRCDIR      := src/
ROOTDIR     := .
ROOTBINDIR  := bin/

CU_DEPS     := $(addprefix $(SRCDIR)/, $(CUHEADERS)) $(COMMON_DEPS)
C_DEPS      := $(addprefix $(SRCDIR)/, $(CCHEADERS))

include common.mk

# Misc. special control flags
CXXFLAGS    += -DVERBOSE

ifeq ($(nounicode),1)
  CXXFLAGS += -DNOUNICODE
endif