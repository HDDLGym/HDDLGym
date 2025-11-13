FROM nvcr.io/nvidia/jax:23.10-py3

# Default workdir
WORKDIR /home/workdir

# Clone the JaxMARL repository
RUN git clone https://github.com/FLAIROx/JaxMARL.git

# Change working directory to the cloned repository
WORKDIR /home/workdir/JaxMARL

# Install JaxMARL and all the requirements system-wide
RUN pip install -e .

# Install additional Python packages
RUN pip install imageio
RUN pip install -U dm-haiku
RUN pip install torch torchvision
RUN pip install gymnasium


# Install tmux
RUN apt-get update && \
    apt-get install -y tmux

# Disabling preallocation
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
# Safety measures
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Uncomment below if you want jupyter 
# RUN pip install jupyterlab

# For secrets and debug
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""
RUN git config --global --add safe.directory /home/workdir

# Set PYTHONPATH
ENV PYTHONPATH="/home/local_workdir/JaxMARL:/home/workdir/JaxMARL:$PYTHONPATH"

# Copy local directory to ensure any additional files are available
COPY . /home/local_workdir/JaxMARL/
WORKDIR /home/local_workdir/JaxMARL