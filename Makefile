################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= ai
# Cuda source files (compiled with cudacc)
CUFILES		:= state.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 


################################################################################
# Rules and targets

include ../../common/common.mk
