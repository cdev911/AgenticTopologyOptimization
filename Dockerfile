FROM dolfinx/dolfinx@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/usr/local/dolfinx-real/lib/python3.12/dist-packages:/usr/local/dolfinx-complex/lib/python3.12/dist-packages:${PYTHONPATH}
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip=24.0+dfsg-1ubuntu1.3 \
    python3-venv=3.12.3-0ubuntu2.1 \
    libgl1-mesa-dev=25.2.8-0ubuntu0.24.04.2 \
    xvfb=2:21.1.12-1ubuntu1.6 && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace
RUN pip3 install --no-cache-dir \
    --constraint /workspace/requirements/runtime.lock \
    -e /workspace

CMD ["bash"]
