# =============================================================================
#  Megatron-LM training image
#  PyTorch 2.11 + CUDA 12.8 + TransformerEngine 2.16.1 + Apex + Flash-Attn 4
# =============================================================================
FROM nvidia/cuda:12.8.1-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
        software-properties-common curl wget git \
        build-essential g++ ninja-build \
        libopenmpi-dev openmpi-bin openmpi-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# ── Make python3.12 the default python / python3 / python3-config ─────────────
# Also fixes the Makefile suffix issue (python3-config → 3.12-config)
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3-config python3-config \
                           /usr/bin/python3.12-config 1

# ── pip ───────────────────────────────────────────────────────────────────────
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && pip install --no-cache-dir --upgrade pip wheel setuptools

# ── PyTorch 2.11.0 with CUDA 12.8 ────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Core build tools ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir pybind11 psutil

# ── Megatron-LM (fork, scaling_laws branch) ──────────────────────────────────
WORKDIR /workspace
RUN git clone https://github.com/martin-kukla/Megatron-LM.git \
    && cd Megatron-LM \
    && git checkout scaling_laws \
    && pip install --no-cache-dir -e .

# ── Pre-compile helpers_cpp with the correct Python 3.12 suffix ───────────────
# Calling g++ directly bypasses the Makefile's python3-config ambiguity.
RUN cd /workspace/Megatron-LM/megatron/core/datasets && \
    LIBEXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))") && \
    g++ -O3 -Wall -shared -std=c++17 -fPIC -fdiagnostics-color \
        $(python3 -m pybind11 --includes) \
        helpers.cpp -o "helpers_cpp${LIBEXT}"

# ── Transformer Engine 2.16.1 ─────────────────────────────────────────────────
# CPATH provides the NCCL headers bundled with the PyTorch NCCL wheel.
RUN NCCL_INCLUDE=$(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia/nccl/include && \
    CPATH="${NCCL_INCLUDE}" \
    pip install --no-cache-dir \
        "transformer_engine[pytorch]==2.16.1" \
        --no-build-isolation

# ── Apex ──────────────────────────────────────────────────────────────────────
# NOTE: This step compiles CUDA kernels — expect 30-60 min build time.
RUN cd /tmp && \
    git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" \
        . && \
    cd /tmp && rm -rf apex

# ── Other utilities ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    tensorboard \
    transformers \
    "huggingface_hub[cli]" \
    hf_transfer \
    simpy

# ── Default environment variables ─────────────────────────────────────────────
ENV NCCL_NET=Socket
ENV NCCL_IB_DISABLE=1
ENV NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1
ENV NVTE_FUSED_ATTN=0

WORKDIR /workspace/Megatron-LM
