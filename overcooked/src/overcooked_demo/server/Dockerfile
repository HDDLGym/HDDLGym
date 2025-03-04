FROM python:3.9

ARG BUILD_ENV
ARG OVERCOOKED_BRANCH
ARG GRAPHICS
ARG GITHUB_TOKEN
ARG REPO_URL
ARG WANDB_API_KEY

WORKDIR /app

# Copy the requirements file into the image
COPY ./requirements.txt ./requirements.txt

# Install Python dependencies with increased timeout and alternative index
RUN pip install --upgrade pip \
    && PIP_DEFAULT_TIMEOUT=300 \
    && pip config set global.timeout 300 \
    && pip config set global.index-url https://pypi.org/simple \
    && pip install -r requirements.txt \
    && if [ "$BUILD_ENV" = "production" ] ; then pip install eventlet ; fi


RUN git clone --recursive https://${GITHUB_TOKEN}@${REPO_URL} --branch main --single-branch /temp_overcooked_ai \
    && mkdir -p /overcooked_ai \
    && cp -a /temp_overcooked_ai/overcooked/. /overcooked_ai/ \
    && echo "import os; DATA_DIR=os.path.abspath('.')" >> /overcooked_ai/src/human_aware_rl/data_dir.py \
    && rm -rf /temp_overcooked_ai \
    && pip install -e '/overcooked_ai[harl]'

    # Modify the sources.list to use the Debian archive and install system packages in one step
    RUN echo "deb http://archive.debian.org/debian stretch main contrib non-free" > /etc/apt/sources.list \
    && echo "deb-src http://archive.debian.org/debian stretch main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb http://archive.debian.org/debian-security stretch/updates main" >> /etc/apt/sources.list \
    && echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/10no-check-valid-until \
    && apt-get -y update \
    && apt-get install -y libgl1-mesa-dev
    
    RUN pip install torch
    
# overwrite layouts with local ones
COPY ./layouts /overcooked_ai/src/overcooked_ai_py/data/layouts
COPY ./hddl /app/hddl
    
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""


# Copy over remaining files
COPY ./static ./static
COPY ./*.py ./
COPY ./graphics/$GRAPHICS ./static/js/graphics.js
COPY ./config.json ./config.json

# Set environment variables that will be used by app.py
ENV HOST=0.0.0.0
ENV PORT=5000
ENV CONF_PATH=config.json

# Expose the port and specify the command to run on container start
EXPOSE 5000
CMD ["python", "-u", "app.py"]
# CMD ["/bin/bash"]