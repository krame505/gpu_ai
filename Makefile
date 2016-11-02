################################################################################
#
# Build script for project
#
################################################################################

# Enable C++ 11
CXXFLAGS+=-std=gnu++0x

# Add source files here
EXECUTABLE	:= ai
# Cuda source files (compiled with cudacc)
CUFILES		:= state.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= mcts.cpp


################################################################################
# Rules and targets

include ../../common/common.mk
