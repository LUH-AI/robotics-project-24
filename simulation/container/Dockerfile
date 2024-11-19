FROM ubuntu:22.04
LABEL version="0.1"
LABEL description="This is a custom Docker Image for running IsaacLab with support for the Go2 Unitree robot"

# Disable Prompt During Package Installation
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y

# Install basic utilities
RUN apt-get install -y curl git wget neovim tmux zsh cmake build-essential ncurses-term

# Install Python 3.10 (needed by IsaacSim)
RUN --mount=type=cache,target=/root/.cache \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && \
  apt-get install -y python3.10 python3-pip python-is-python3 && \
  pip install --upgrade pip

# Install IsaacSIM
RUN --mount=type=cache,target=/root/.cache \
  pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && \
  pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com && \
  pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com

# Install IsaacLab
## This solves a problem with https cloning of large repositories
RUN --mount=type=cache,target=/root/.cache \
  git config --global http.postBuffer 524288000 && \
  git clone https://github.com/isaac-sim/IsaacLab.git && \
  cd IsaacLab && \
  echo "Yes" | ./isaaclab.sh --install
