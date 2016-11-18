CUDA_INSTALL_PATH := /usr/local/cuda-8.0
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
NVCCFLAGS := -gencode arch=compute_20,code=sm_20 -g -G -Wno-deprecated-gpu-targets

CXX := /opt/rh/devtoolset-4/root/usr/bin/g++
CXXFLAGS := -std=gnu++11 -g -Wall

EXECUTABLE	:= run_ai
# Cuda source files (compiled with cudacc)
CUFILES		:= $(wildcard *.cu)
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= $(wildcard *.cpp)
# Header files / anything that should trigger a full rebuild
#C_DEPS          := colors.h mcts.hpp playout.hpp state.hpp player.cpp
CPP_OBJS          := $(patsubst %.cpp,%.o,$(CCFILES))
CU_OBJS           := $(patsubst %.cu,%.o,$(CUFILES))
LIBS := -L"$(CUDA_INSTALL_PATH)/lib64" -lcudart -lcudadevrt -lcurand
INCD := -I"$(CUDA_INSTALL_PATH)/include" -I.

all : $(EXECUTABLE)

%.o : %.cu
	@$(NVCC) $(NVCCFLAGS) -dc $(INCD) -o $@ $<

%.o : %.cpp
	@$(CXX) -c $(CXXFLAGS) $(INCD) -o $@ $<

device.o : $(CU_OBJS)
	@$(NVCC) $(NVCCFLAGS) -dlink $(CU_OBJS) -o device.o

$(EXECUTABLE) : $(CPP_OBJS) $(CU_OBJS) device.o
	@$(CXX) -o $(EXECUTABLE) device.o $(CPP_OBJS) $(CU_OBJS) $(LIBS)

clean :
	@rm -f $(EXECUTABLE) *.o
