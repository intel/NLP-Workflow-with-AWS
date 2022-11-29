# Copyright (C) 2022 Intel Corporation
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
 
# http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
 

# SPDX-License-Identifier: Apache-2.0

FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.12.1-cpu-py38-ubuntu20.04-ec2

#Install necessary packages
RUN apt-get update && \
    apt-get install --no-install-recommends wget=1.20.3-1ubuntu2 -y && \
    apt-get install --no-install-recommends python3.8-dev=3.8.10-0ubuntu1~20.04.5 -y && \
    apt-get install --no-install-recommends python3-pip=20.0.2-5ubuntu1.6 -y && \
    apt-get install --no-install-recommends libgl1=1.3.2-1~ubuntu0.20.04.2 -y &&\
    apt-get install --no-install-recommends python3-opencv=4.2.0+dfsg-5 -y && \
    apt-get install --no-install-recommends build-essential=12.8ubuntu1.1 -y && \
    apt-get install --no-install-recommends libtool=2.4.6-14 -y &&\
    apt-get install --no-install-recommends autoconf=2.69-11.1 -y &&\
    apt-get install --no-install-recommends libssl-dev=1.1.1f-1ubuntu2.16 -y \
    && rm -rf /var/lib/apt/lists/*

#Update the PyTorch
RUN pip install --no-cache-dir --upgrade pip==22.3 && \
    pip install --no-cache-dir torch==1.12.1 && \
    pip install --no-cache-dir cerberus==1.3.4 && \
    pip install --no-cache-dir flatbuffers==2.0 && \
    pip install --no-cache-dir h5py==3.7.0 && \
    pip install --no-cache-dir numpy==1.23.1 && \
    pip install --no-cache-dir packaging==21.3 && \
    pip install --no-cache-dir protobuf==3.20.1 && \
    pip install --no-cache-dir sympy==1.10.1 && \
    pip install --no-cache-dir setuptools==63.2.0 && \
#Install the environment
    pip install --no-cache-dir intel-extension-for-pytorch==1.12.100 && \
    pip install --no-cache-dir transformers==4.21.1 && \
    pip install --no-cache-dir datasets==2.4.0 && \
    pip install --no-cache-dir pandas==1.2.5 && \
    pip install --no-cache-dir PyYAML==5.4.1 && \
    pip install --no-cache-dir neural-compressor==1.14.1

#Specify the quantization behavior
COPY inc_quantization.py /opt/ml/code/inc_quantization.py
COPY inc_config.yaml /opt/ml/code/inc_config.yaml
CMD ["python", "/opt/ml/code/inc_quantization.py"]