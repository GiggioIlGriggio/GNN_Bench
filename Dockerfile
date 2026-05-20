# Use the latest stable PyTorch runtime as the base
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Copy and install Python dependencies.
# nvidia-cu12 / cuda-* packages are intentionally excluded from requirements.txt
# because the base image already provides the CUDA 12.4 system toolkit; installing
# pip-distributed CUDA 12.8+ libs on top would cause shared-library conflicts.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch Geometric and its compiled extensions.
# torch_geometric itself is pure-Python; the optional compiled extensions
# (scatter, sparse, cluster, spline-conv) must be fetched from the PyG wheel
# index built for the exact PyTorch + CUDA version in the base image.
RUN pip install --no-cache-dir torch_geometric && \
    pip install --no-cache-dir \
        torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

CMD ["python"]