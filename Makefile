################################################################################
#
# Build script for project
#
################################################################################

# Using a newer version of CUDA than the default
CUDA_INSTALL_PATH ?= /usr/local/cuda-8.0

EXECUTABLE  := run_ai

# CUDA source files (compiled with cudacc)
CUFILES	    := state.cu playout.cu genMovesTest.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES	    := state.cpp playout.cpp mcts.cpp player.cpp driver.cpp
# Header files included by any of CUFILES
CUHEADERS   := playout.hpp state.hpp genMovesTest.hpp
# Header files included by any of CCFILES
CCHEADERS   := colors.h mcts.hpp playout.hpp state.hpp player.hpp

SRCDIR      := src
ROOTDIR     := .

PARENTBINDIR := bin
PARENTOBJDIR := obj

ifeq ($(dbg),1)
  ROOTBINDIR  := $(PARENTBINDIR)/debug
  OBJDIR      := $(PARENTOBJDIR)/debug
else
  ROOTBINDIR  := $(PARENTBINDIR)/release
  OBJDIR      := $(PARENTOBJDIR)/release
endif

CU_DEPS     := $(addprefix $(SRCDIR)/, $(CUHEADERS)) Makefile
C_DEPS      := $(addprefix $(SRCDIR)/, $(CCHEADERS)) Makefile

CUOBJS      := $(patsubst %.cu, $(OBJDIR)/%.cu.o, $(CUFILES))
CCOBJS      := $(patsubst %.cpp, $(OBJDIR)/%.cpp.o, $(CCFILES))

NVCC        := $(CUDA_INSTALL_PATH)/bin/nvcc
NVCCFLAGS   := --generate-code arch=compute_20,code=sm_20 --generate-code arch=compute_30,code=sm_30 -Wno-deprecated-gpu-targets -m64 -DUNIX --compiler-options -fno-strict-aliasing -I"$(CUDA_INSTALL_PATH)/include"

CXX         := g++
CXXFLAGS    := -fopenmp -fno-strict-aliasing -m64 -std=gnu++11 -Wall -Wextra -DVERBOSE -DUNIX -I"$(CUDA_INSTALL_PATH)/include"

LIB         += -lgomp -L"$(CUDA_INSTALL_PATH)/lib64" -lcuda -lcudart

ifeq ($(dbg),1)
  CXXFLAGS  += -g3 -ggdb
  NVCCFLAGS += -g -G
else
  CXXFLAGS  += -O3 -DNDEBUG
  NVCCFLAGS += -O3 -DNDEBUG
endif

ifeq ($(nounicode),1)
  CXXFLAGS  += -DNOUNICODE
endif

ifeq ($(verbose),1)
  V         := 
else
  V         := @
endif

.PHONY : all clean clobber

all : $(ROOTBINDIR)/$(EXECUTABLE)

$(OBJDIR)/%.cu.o : $(SRCDIR)/%.cu $(CU_DEPS) | $(OBJDIR)
	$(V)$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

$(OBJDIR)/%.cpp.o : $(SRCDIR)/%.cpp $(C_DEPS) | $(OBJDIR)
	$(V)$(CXX) -c $(CXXFLAGS) -o $@ $<

$(OBJDIR)/device.o : $(CUOBJS) | $(OBJDIR)
	$(V)$(NVCC) $(NVCCFLAGS) -dlink $(CUOBJS) -o $@

$(ROOTBINDIR)/$(EXECUTABLE) : $(CUOBJS) $(CCOBJS) $(OBJDIR)/device.o | $(ROOTBINDIR)
	$(V)$(CXX) -o $@ $(CUOBJS) $(CCOBJS) $(OBJDIR)/device.o $(LIB)

$(OBJDIR) :
	$(V)mkdir -p $(OBJDIR)

$(ROOTBINDIR) :
	$(V)mkdir -p $(ROOTBINDIR)

clean :
	$(V)rm -f $(ROOTBINDIR)/$(EXECUTABLE) $(CUOBJS) $(CCOBJS) $(OBJDIR)/device.o

clobber : clean
	$(V)rm -rf $(PARENTBINDIR) $(PARENTOBJDIR)
