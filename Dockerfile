ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.08-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /code
ADD . .
