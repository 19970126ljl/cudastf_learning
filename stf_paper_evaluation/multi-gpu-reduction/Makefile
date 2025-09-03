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

# Keep flags (disable by default, enable with make KEEP=1)
KEEP ?= 0
ifeq ($(KEEP), 1)
    KEEP_FLAGS = --keep --keep-dir keep_dir -Xptxas -v
else
    KEEP_FLAGS =
endif

NVCCFLAGS += $(KEEP_FLAGS)
# Compiler flags for maxregcount kernel
NVCCFLAGS_MAXREG = $(NVCCFLAGS) -maxrregcount=32



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
MAXREG_KERNEL_OBJ = stf_launch_maxregcount.o
MAXREG_KERNEL_SRC = kernels/stf_launch_maxregcount.cu


# Default target
all: $(TARGET)

# Rule to build the maxregcount kernel object
$(MAXREG_KERNEL_OBJ): $(MAXREG_KERNEL_SRC)
	@echo "Compiling maxregcount kernel..."
	$(NVCC) $(NVCCFLAGS_MAXREG) $(INCLUDES) -c $(MAXREG_KERNEL_SRC) -o $(MAXREG_KERNEL_OBJ)

# Rule to build the executable
$(TARGET): $(SOURCE) $(MAXREG_KERNEL_OBJ)
	@echo "Compiling and linking $(TARGET)..."
	$(NVCC) $(NVCCFLAGS) $(KEEP_FLAGS) $(INCLUDES) $(SOURCE) $(MAXREG_KERNEL_OBJ) -o $(TARGET) $(LDFLAGS) $(LIBS)
	@echo "Compilation successful!"
	@echo "Run the benchmark with: $./$(TARGET)"

# Keep target - compile with intermediate files preserved
keep:
	@echo "Compiling $(TARGET) with intermediate files preserved..."
	$(MAKE) KEEP=1 all

# Clean target
clean:
	@echo "Cleaning..."
	rm -f $(TARGET) $(MAXREG_KERNEL_OBJ)
	rm -f reduction_benchmark_v1 reduction_benchmark_v2
	rm -f *.sass *.ptx *.cubin *.ii *_intermediate
	rm -rf intermediate_v1/ intermediate_v2/
	rm -f *_disasm.txt *_sass*.txt
	rm -f *.csv *.png
	rm -rf perf_out/
	rm -rf keep_dir/
	@echo "Clean complete."

# Run target
run: $(TARGET)
	@echo "Running $(TARGET)..."
	./$(TARGET)

# Phony targets
.PHONY: all clean run keep
