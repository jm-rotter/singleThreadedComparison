NVCC = nvcc

NVCCFLAGS = -std=c++17 -Iinclude -O2 -Xcompiler "-Wall -Wextra"

TARGET = stp

SRCS = src/cpu_matmul.cpp src/utils.cpp src/gpu_matmul.cu src/main.cu

CPP_OBJS := $(SRCS:.cpp=.o) 
OBJS := $(CPP_OBJS:.cu=.o)

all: $(TARGET)

$(TARGET): $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) -o $@ $^ -Iinclude 

%.o: %.cpp %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean: rm -f $(CPU_OBJS) $(GPU_OBJS) $(TARGET)
