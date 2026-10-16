# MC3 solution image. MUST stay on the mandated ROCm base (checked by layer
# identity at evaluation) and under 60 GiB uncompressed.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
# Guard: pip must not silently swap the ROCm torch for a CUDA build
# (documented Match 3 pitfall). Fails the build loudly instead of shipping
# a silently CPU-only image, which would score zero on the VRAM floor check.
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && python3 -c "import torch; v=torch.__version__; assert 'rocm' in v.lower(), f'FATAL: torch lost its ROCm build during pip install: {v}'"

# Model weights are baked in at BUILD time (no network at evaluation).
# Run on the AMD GPU build machine; needs HF access at build time.
COPY download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py && rm /tmp/download_models.py

COPY app/ /app/
RUN mkdir -p /app/corpus /app/output /app/index

# No service, no port. The harness execs into the running container;
# our GPU daemon is started by the --index pass and kept on a unix socket.
CMD ["sleep", "infinity"]
