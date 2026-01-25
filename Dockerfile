# Use the specific NVIDIA CUDA 13.1.1 image with cuDNN and Ubuntu 24.04
FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python, pip, and build tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libcupti-dev \
    build-essential

# Set the working directory
WORKDIR /home

# Copy requirements
COPY requirements.txt .

# Installs the packages in the requirements.txt and additional CUDA packages.
RUN pip3 install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --break-system-packages

# Copy the rest of the project
COPY . .

CMD ["echo", "BUILD IS SUCCESSFUL!!!!!"]