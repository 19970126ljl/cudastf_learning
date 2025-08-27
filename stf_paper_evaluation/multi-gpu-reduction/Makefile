# Makefile for the reduction benchmark

# Compiler
NVCC = /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/bin/nvcc

# Compiler flags
NVCCFLAGS = -std=c++17 -expt-relaxed-constexpr --extended-lambda -arch=native -lcuda -lineinfo

# Debug flag (disable by default, enable with make DEBUG=1)
DEBUG ?= 0
ifeq ($(DEBUG), 1)
    NVCCFLAGS += -g -G -O0
else
    NVCCFLAGS += -O3
endif

# Linker flags
LDFLAGS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib
LIBS =

# Include paths
INCLUDES = -I../../cccl/libcudacxx/include \
           -I../../cccl/cudax/include \
           -I../../cccl/cub \
           -I../../cccl/thrust

# Target executable
TARGET = reduction_benchmark

# Source file
SOURCE = reduction_benchmark.cu

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(SOURCE)
	@echo "Compiling $(TARGET)..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SOURCE) -o $(TARGET) $(LDFLAGS) $(LIBS)
	@echo "Compilation successful!"
	@echo "Run the benchmark with: $./$(TARGET)"

# Clean target
clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	rm -f reduction_benchmark_v1 reduction_benchmark_v2
	rm -f *.sass *.ptx *.cubin *.ii *_intermediate
	rm -rf intermediate_v1/ intermediate_v2/
	rm -f *_disasm.txt *_sass*.txt
	rm -f *.csv *.png
	rm -f perf_out/
	@echo "Clean complete."

# Run target
run: $(TARGET)
	@echo "Running $(TARGET)..."
	./$(TARGET)

# Phony targets
.PHONY: all clean run