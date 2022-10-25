ARG FROM_IMAGE_NAME=pytorch/pytorch:latest
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/torchrec_dlrm
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
