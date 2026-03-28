CXX ?= g++
NVCC ?= nvcc

TARGET := main
OBJDIR := obj
SRCDIR := src
INCDIR := include

CUDA_HOME ?= $(shell dirname $$(dirname $$(readlink -f $$(command -v $(NVCC)))))
CUDA_INC_DIR ?= $(firstword $(wildcard $(CUDA_HOME)/include /usr/include))
CUDA_LIB_DIR ?= $(firstword $(wildcard $(CUDA_HOME)/lib64 $(CUDA_HOME)/targets/x86_64-linux/lib /usr/lib/x86_64-linux-gnu))

CXX_STD ?= c++17
DEPFLAGS := -MMD -MP
CXXFLAGS := -std=$(CXX_STD) -O3 -Wall -march=native -mavx2 -mfma -mno-avx512f -fopenmp $(DEPFLAGS) -I$(INCDIR) -I$(CUDA_INC_DIR)
CUDA_ARCH ?= sm_70
NVCC_HOST_FLAGS := -O3 -Wall -fopenmp
NVCCFLAGS := -std=$(CXX_STD) -arch=$(CUDA_ARCH) $(DEPFLAGS) $(foreach option,$(NVCC_HOST_FLAGS),-Xcompiler=$(option)) -I$(INCDIR) -I$(CUDA_INC_DIR)
LDFLAGS := -pthread -L$(CUDA_LIB_DIR)
LDLIBS := -lstdc++ -lcudart -lm

OBJECTS := $(OBJDIR)/main.o $(OBJDIR)/app_options.o $(OBJDIR)/generation.o $(OBJDIR)/model.o $(OBJDIR)/tensor.o $(OBJDIR)/layer.o $(OBJDIR)/util.o $(OBJDIR)/safetensors_loader.o $(OBJDIR)/config.o
DEPFILES := $(OBJECTS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS) $(LDFLAGS) $(LDLIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | create_obj
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | create_obj
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

create_obj:
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(TARGET) $(OBJECTS) $(DEPFILES)

run:
	sh ./run.sh

-include $(DEPFILES)

.PHONY: all clean run create_obj
