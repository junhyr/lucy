# Lucy 2.0 PoC — RunPod H100 Environment
# Base: PyTorch 2.6 + CUDA 12.4 (matches StreamDiffusionV2 requirements)
FROM runpod/pytorch:2.6.0-py3.11-cuda12.4.1-cudnn9-devel

LABEL maintainer="lucy-poc"
LABEL description="StreamDiffusionV2 inference benchmarking on H100"

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="9.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone repos
RUN git clone --depth 1 https://github.com/cumulo-autumn/StreamDiffusionV2.git repos/StreamDiffusionV2 && \
    git clone --depth 1 https://github.com/chengzeyi/stable-fast.git repos/stable-fast && \
    git clone --depth 1 https://github.com/SageSR/TurboDiffusion.git repos/TurboDiffusion && \
    git clone --depth 1 https://github.com/NVlabs/rcm.git repos/rcm

# Install core Python dependencies
COPY scripts/setup/requirements_poc.txt /workspace/requirements_poc.txt
RUN pip install --no-cache-dir -r /workspace/requirements_poc.txt

# Install flash-attn (build from source for H100 SM90)
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Install StreamDiffusionV2
RUN cd repos/StreamDiffusionV2 && pip install -e . 2>/dev/null || true

# Install stable-fast (with CUTLASS submodule)
RUN cd repos/stable-fast && \
    git submodule update --init --recursive && \
    pip install -e . 2>/dev/null || true

# Create output dirs
RUN mkdir -p /workspace/{checkpoints,results,network-volume}

# Copy PoC scripts
COPY . /workspace/lucy-poc/

WORKDIR /workspace/lucy-poc
