# Set CUDA Environment Variables for NVIDIA HPC SDK 2025 via .env

# export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/cuda/11.8
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/cuda/12.8

export PATH=${CUDA_HOME}/bin:${PATH}

export CUDACXX=${CUDA_HOME}/bin/nvcc

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

export CUDA_VISIBLE_DEVICES=3
