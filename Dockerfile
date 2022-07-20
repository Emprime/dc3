FROM tensorflow/tensorflow:1.14.0-gpu-py3

# Fix to old cuda key in docker image (see https://github.com/NVIDIA/nvidia-docker/issues/1632)
RUN apt-key del 7fa2af80
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

ARG DEBIAN_FRONTEND=noninteractive

# Update dependencies and install required packages
# RUN apt-get -y upgrade
RUN apt-get update
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx git iproute2 pylint
RUN python3 -m pip install --upgrade pip
RUN git config --global --add safe.directory /src
COPY requirements.txt /tmp/
RUN pip3 install --requirement /tmp/requirements.txt



