# Makefile for the reduction benchmark

# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -std=c++17 -expt-relaxed-constexpr --extended-lambda -arch=native -lcuda -lineinfo

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
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(SOURCE) -o $(TARGET)
	@echo "Compilation successful!"
	@echo "Run the benchmark with: ./$(TARGET)"

# Clean target
clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	@echo "Clean complete."

# Run target
run: $(TARGET)
	@echo "Running $(TARGET)..."
	./$(TARGET)

# Phony targets
.PHONY: all clean run