#!/usr/bin/env bash

# Install NCCL
sudo ln -sf /usr/bin/python3 /usr/bin/python
python3 -c "from cupy.cuda import nccl"
python3 -m cupyx.tools.install_library --library nccl --cuda 11.x
nvidia-smi
wget https://developer.download.nvidia.com/compute/redist/nccl/v2.15.1/nccl_2.15.1-1+cuda11.8_x86_64.txz
tar Jxvf nccl_2.15.1-1+cuda11.8_x86_64.txz
mkdir .cupy/cuda_lib/11.x/nccl/2.15.1/
cp -R nccl_2.15.1-1+cuda11.8_x86_64/lib .cupy/cuda_lib/11.x/nccl/2.15.1/
cp -R nccl_2.15.1-1+cuda11.8_x86_64/include .cupy/cuda_lib/11.x/nccl/2.15.1/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.cupy/cuda_lib/11.x/nccl/2.15.1/lib/


# Download cudnn-local-repo-ubuntu2004-8.8.0.121_1.0-1_amd64.deb
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.8.0/local_installers/11.8/cudnn-local-repo-ubuntu2004-8.8.0.121_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.8.0.121_1.0-1_amd64.deb  
sudo cp /var/cudnn-local-repo-ubuntu2004-8.8.0.121/cudnn-local-B70907B4-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt list libcudnn8
sudo apt-get install libcudnn8=8.8.0.121-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.8.0.121-1+cuda11.8
sudo apt-get install libcudnn8-samples=8.8.0.121-1+cuda11.8
dpkg -l | grep cuda

# Install GCC
sudo apt install gcc-7 g++-7 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 1
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 1

# Select the working version of protobuf/grpcio
pip3 install protobuf==3.20.3 grpcio==1.43.0

# Checkout Alpa
git clone --recursive https://github.com/alpa-projects/alpa.git
cd alpa/
pip3 install -e "."

# Build jaxlib
cd build_jaxlib/
rm -rf ~/.cache/bazel/
python3 build/build.py --enable_cuda --dev_install --bazel_options=--override_repository=org_tensorflow=$(pwd)/../third_party/tensorflow-alpa
cd dist
pip3 install -e .
cd
# To fix missing alpa module error
cd alpa/
pip3 install -e "."

# Install additional packages
pip3 install ray[default]==2.1.0 # also 2.0.0 works
pip3 install pydantic==1.10.13
pip3 install torch==2.0.1 torchvision==0.15.2 
pip3 install transformers
pip3 install tensorflow-gpu==2.9 tensorflow==2.9 protobuf==3.20.0 datasets # previous 2.8 was slow for large datasets
pip3 install markupsafe==2.0.1 tokenizers==0.20.3

echo "👉 Done!"