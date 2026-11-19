FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5 ninja==1.11.1.1

WORKDIR /app

COPY . /tests/
