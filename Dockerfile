# # Base container that includes all dependencies but not the actual repo
# # Updated from templates in the [softlearning (SAC) library](https://github.com/rail-berkeley/softlearning)

# FROM nvidia/cudagl:11.4.2-base-ubuntu20.04 as base
# # FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04 as base
# # ARCH and CUDA are specified again because the FROM directive resets ARGs
# # (but their default value is retained if set previously)

# SHELL ["/bin/bash", "-c"]


# ENV DEBIAN_FRONTEND="noninteractive"
# # See http://bugs.python.org/issue19846
# ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# ENV PATH /opt/conda/bin:$PATH

# # install anaconda
# RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
#     libglib2.0-0 libxext6 libsm6 libxrender1 \
#     git mercurial subversion
    

# RUN apt-get install -y wget bzip2 ca-certificates git vim
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
# 		build-essential \
# 		cmake \
#         vim \
#         ffmpeg \
#         unzip swig libfreetype6-dev \
#         libosmesa6-dev patchelf ffmpeg \
#         freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglew1.6-dev mesa-utils

# # Not sure what this is fixing
# # COPY ./files/Xdummy /usr/local/bin/Xdummy
# # RUN chmod +x /usr/local/bin/Xdummy
        
# ENV PATH /opt/conda/bin:$PATH
# RUN wget --quiet https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#     /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
#     rm /tmp/miniconda.sh && \
#     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

# RUN conda update -y --name base conda && conda clean --all -y

# RUN conda create --name roble python=3.8 pip
# RUN echo "source activate roble" >> ~/.bashrc
# ## Make it so you can install things to the correct version of pip
# ENV PATH /opt/conda/envs/roble/bin:$PATH
# RUN source activate roble

# # make sure your domain is accepted and can be pulled from in the container
# # RUN touch /root/.ssh/known_hosts
# RUN mkdir /root/.ssh
# RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# ## You can change this to a folder you prefer.
# RUN mkdir /root/playground

# RUN ls
# ## Install the requirements for your learning code.
# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt

# # RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# # COPY you code to the docker image here.
# # e.g.
# # COPY tools/openAIGym.py main.py
# ADD conf conf
# # ADD hw1 hw1
# # ADD tools tools
# # COPY run_hw1_bc.py run_hw1_bc.py
# ADD hw7 hw7
# COPY run_hw7_sim2real.py run_hw7_sim2real.py
# COPY vanilla_env.py vanilla_env.py

# ## Install pytorch and cuda
# RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# # Check that the sime loads and pre-build mujoco in the docker image. Better to catch these errors here.
# # RUN python -c "import gym; env = gym.make('Ant-v2'); print(env)"
# # COPY you code to the docker image here.
# # e.g.
# # ADD conf conf
# # ADD hw1 hw1
# # ADD hw2 hw2
# # ADD hw3 hw3
# # ADD hw4 hw4
# # ADD hw5 hw5
# # ADD hw6 hw6
# # ADD tools tools
# # COPY run_hw1_bc.py run_hw1_bc.py
# # COPY run_hw2_mb.py run_hw2_mb.py
# # COPY run_hw3_ql.py run_hw3_ql.py
# # COPY run_hw4_gc.py run_hw4_gc.py
# # COPY run_hw5_expl.py run_hw5_expl.py
# # COPY run_hw6_sim2real.py run_hw6_sim2real.py

# ## Check the file were copied
# RUN ls

############


# Use the specified NVIDIA CUDA base image
FROM nvidia/cuda:12.0.1-devel-ubuntu20.04 as base

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND="noninteractive"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install system packages
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion build-essential cmake \
    vim ffmpeg unzip swig libfreetype6-dev \
    libosmesa6-dev patchelf freeglut3-dev \
    build-essential libx11-dev libxmu-dev libxi-dev \
    libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev \
    libglew1.6-dev mesa-utils

# Install Anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

# Copy the environment file into the Docker image
COPY environment.yml /tmp/environment.yml

# Create Conda environment from the environment.yml file
RUN conda env create -f /tmp/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "cloudspace", "/bin/bash", "-c"]

# Set up SSH
RUN mkdir /root/.ssh && ssh-keyscan github.com >> /root/.ssh/known_hosts

# Copy project files
COPY hw7 /root/playground/hw7
COPY run_hw7_sim2real.py /root/playground/run_hw7_sim2real.py
COPY requirements.txt /root/playground/requirements.txt

# Additional Python packages installation if needed
RUN pip install -r /root/playground/requirements.txt

# Configure container startup
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cloudspace", "bash", "-c", "while true; do sleep 30; done;"]

RUN ls