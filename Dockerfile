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

FROM ubuntu:20.04

#Install necessary packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata=2022f-0ubuntu0.20.04.1 --no-install-recommends && rm -rf /var/lib/apt/lists/*
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget=1.20.3-1ubuntu2 \
         nginx=1.18.0-0ubuntu1.4 \
         cmake=3.16.3-1ubuntu1.20.04.1 \
         software-properties-common=0.99.9.8 \
         pkg-config=0.29.1-0ubuntu4 \
         python3.8-dev=3.8.10-0ubuntu1~20.04.5 \
         build-essential=12.8ubuntu1.1 \
         ca-certificates=20211016~20.04.1 \
         gnupg=2.2.19-3ubuntu2.2 \
         python3-pip=20.0.2-5ubuntu1.6 \
         libgl1=1.3.2-1~ubuntu0.20.04.2 \
         python3-opencv=4.2.0+dfsg-5 \
    && rm -rf /var/lib/apt/lists/*

#Enable Intel optimized Tensorflow
ENV TF_ENABLE_ONEDNN_OPTS=1

#Install the environment
#RUN pip install --no-cache-dir --upgrade pip && \
RUN pip install --no-cache-dir boto3==1.24.15 &&\
    pip install --no-cache-dir sagemaker==2.96.0 &&\
    pip install --no-cache-dir tensorflow-cpu==2.9.1 &&\
    pip install --no-cache-dir transformers==4.19.2 &&\
    pip install --no-cache-dir datasets==2.3.2 &&\
    pip install --no-cache-dir pandas==1.2.5 &&\
    pip install --no-cache-dir protobuf==3.20.0 &&\
    pip install --no-cache-dir tensorflow-hub==0.12.0 &&\
    pip install --no-cache-dir notebook==6.4.12 &&\
    pip install --no-cache-dir opencv-python==4.6.0.66 &&\
    pip install --no-cache-dir awscli==1.25.16
    
#Install a new juypter kernel for Intel Neural Compressor
RUN pip install --no-cache-dir virtualenv==20.14.1 && \
    virtualenv intel_neural_compressor_venv && \
    . intel_neural_compressor_venv/bin/activate && \
    pip install --no-cache-dir Cython==0.29.32 && \
    pip install --no-cache-dir tensorflow-cpu==2.9.1 && \
    pip install --no-cache-dir neural-compressor==1.13.1 && \
    pip install --no-cache-dir ipykernel==6.15.0 && \
    python -m ipykernel install --user --name=intel_neural_compressor_kernel

#Copy the codes
COPY /notebooks /root/notebooks
COPY /src /root/src
