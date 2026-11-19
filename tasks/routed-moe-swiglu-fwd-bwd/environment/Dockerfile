FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# ninja is needed by torch.utils.cpp_extension.load_inline.
RUN pip install --no-cache-dir ninja==1.11.1.1

WORKDIR /app

COPY app/ /app/
