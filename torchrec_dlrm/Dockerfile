ARG FROM_IMAGE_NAME=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
FROM ${FROM_IMAGE_NAME}

RUN apt-get -y update && \
    apt-get -y install git

WORKDIR /workspace/torchrec_dlrm
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
