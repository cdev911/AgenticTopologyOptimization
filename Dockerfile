FROM dolfinx/dolfinx:stable

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/dolfinx-complex/lib/python3.12/dist-packages:${PYTHONPATH}
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv libgl1-mesa-dev xvfb && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace
RUN pip3 install --no-cache-dir -e /workspace

CMD ["bash"]
