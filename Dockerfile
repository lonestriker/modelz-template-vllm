# use devel to compile vllm
ARG base=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

FROM ${base}

ARG CONDA_VERSION=py310_23.3.1-0

ENV DEBIAN_FRONTEND=noninteractive LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

RUN apt update && \
    apt install -y --no-install-recommends \
        wget \
        git \
        build-essential \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN set -x && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"; \
        SHA256SUM="aef279d6baea7f67940f16aad17ebe5f6aac97487c7c03466ff01f4819e5a651"; \
    elif [ "${UNAME_M}" = "s390x" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-s390x.sh"; \
        SHA256SUM="ed4f51afc967e921ff5721151f567a4c43c4288ac93ec2393c6238b8c4891de8"; \
    elif [ "${UNAME_M}" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-aarch64.sh"; \
        SHA256SUM="6950c7b1f4f65ce9b87ee1a2d684837771ae7b2e6044e0da9e915d1dee6c924c"; \
    elif [ "${UNAME_M}" = "ppc64le" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-ppc64le.sh"; \
        SHA256SUM="b3de538cd542bc4f5a2f2d2a79386288d6e04f0e1459755f3cefe64763e51d16"; \
    fi && \
    wget "${MINICONDA_URL}" -O miniconda.sh -q && \
    echo "${SHA256SUM} miniconda.sh" > shasum && \
    if [ "${CONDA_VERSION}" != "latest" ]; then sha256sum --check --status shasum; fi && \
    mkdir -p /opt && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh shasum && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

ENV PYTHON_PREFIX=/opt/conda/bin

RUN update-alternatives --install /usr/bin/python python ${PYTHON_PREFIX}/python 1 && \
    update-alternatives --install /usr/bin/python3 python3 ${PYTHON_PREFIX}/python3 1 && \
    update-alternatives --install /usr/bin/pip pip ${PYTHON_PREFIX}/pip 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 ${PYTHON_PREFIX}/pip3 1

# torch should be installed before the vllm to avoid some bugs
RUN pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118 
RUN pip install fschat accelerate ray pandas numpy huggingface_hub
RUN pip install xformers --no-deps
RUN pip install ninja psutil pyarrow sentencepiece transformers fastapi uvicorn[standard] 'pydantic<2'

RUN mkdir -p /workspace

# Use my fork of vllm-release's fork to fix AWQ support, revert to primary vllm repo when AWQ support works
#ARG COMMIT=main
ARG COMMIT=add-mistral
RUN git clone https://github.com/lonestriker/vllm-release.git /workspace/vllm && \
    cd /workspace/vllm && \
    git checkout ${COMMIT} && \
    pip install --no-deps --no-build-isolation .

WORKDIR /workspace

#ARG MODEL=meta-llama/Llama-2-7b-hf
#ARG MODEL_PATH=$MODEL
# download the model
# COPY warmup.py /workspace/warmup.py
# ENV HUGGING_FACE_HUB_TOKEN=
# RUN python /workspace/warmup.py $MODEL

# For a local model instead of downloading
#ARG MODEL_NAME=dolphin-2.0-mistral-7B-AWQ
#ARG MODEL_NAME=CollectiveCognition-v1.1-Mistral-7B-AWQ
#ARG MODEL=models/$MODEL_NAME
#COPY $MODEL $MODEL_NAME
#ARG MODEL_PATH=/workspace/$MODEL_NAME
#ARG QUANT=awq
#ENV MODEL_PATH=$MODEL_PATH
#ENV QUANT=$QUANT

# Run warmup.py if MODEL env var is set to download model at build time
COPY warmup.py /workspace/warmup.py
RUN [ -n "${MODEL:-}" ] && python /workspace/warmup.py $MODEL || true

# Need to use bash command expansion to use env vars in ENTRYPOINT if specifying the model as part of the build
# Sample command to run docker with built-in docker image:
# docker run --gpus '"device=0"' --shm-size 1g -p 8080:8080 lonestriker/vllm:cc-v1.1
# docker run --gpus all --shm-size 1g -p 8080:8080 lonestriker/vllm:dolphin-2.0
#ENTRYPOINT [ "bash", "-c", "python -m vllm.entrypoints.openai.api_server --worker-use-ray --host 0.0.0.0 --port 8080 --gpu-memory-utilization 0.85 --model $MODEL_PATH --quant $QUANT" ]

# Model is specified at runtime below, example docker command:
# docker run --gpus '"device=0"' --shm-size 1g -p 8080:8080 -v /aiml/models:/models lonestriker/vllm:default --model /models/dolphin-2.0-mistral-7B-AWQ --quant awq
ENTRYPOINT [ "python", "-m", "vllm.entrypoints.openai.api_server", "--worker-use-ray", "--host", "0.0.0.0", "--port", "8080", "--gpu-memory-utilization", "0.85" ]
