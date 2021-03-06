################################################################################
#
# Build script for project
#
################################################################################

EXECUTABLE  := run_ai

# CUDA source files (compiled with cudacc)
CUFILES	    := state.cu singlePlayout.cu multiplePlayout.cu heuristicPlayout.cu coarsePlayout.cu heuristic.cu genMovesTest.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES	    := state.cpp playout.cpp heuristicPlayout.cpp mcts.cpp player.cpp driver.cpp
# Header files included by any of CUFILES
CUHEADERS   := playout.hpp state.hpp heuristic.hpp genMovesTest.hpp
# Header files included by any of CCFILES
CCHEADERS   := cxxopts.hpp colors.h mcts.hpp playout.hpp state.hpp player.hpp heuristic.hpp

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

NVCC        := nvcc
NVCCFLAGS   += -m64 -DUNIX -std=c++14 --compiler-options -fno-strict-aliasing

CXX         := g++
CXXFLAGS    += -fopenmp -fno-strict-aliasing -m64 -std=gnu++17 -Wall -Wextra -DVERBOSE -DUNIX

LIB         += -lgomp -lpthread -lcudart

ifeq ($(dbg),1)
  CXXFLAGS  += -g3 -ggdb
  NVCCFLAGS += -g -G -lineinfo
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
