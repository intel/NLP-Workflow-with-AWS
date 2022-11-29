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

RUN apt-get update && \
    apt-get install --no-install-recommends wget=1.20.3-1ubuntu2 -y && \
    apt-get install --no-install-recommends python3.8-dev=3.8.10-0ubuntu1~20.04.5 -y && \
    apt-get install --no-install-recommends python3-pip=20.0.2-5ubuntu1.6 -y && \
    apt-get install --no-install-recommends libgl1=1.3.2-1~ubuntu0.20.04.2 -y &&\
    apt-get install --no-install-recommends python3-opencv=4.2.0+dfsg-5 -y && \
    apt-get install --no-install-recommends openmpi-bin=4.0.3-0ubuntu1 -y && \
    apt-get install --no-install-recommends libopenmpi-dev=4.0.3-0ubuntu1 -y && \
    apt-get install --no-install-recommends openmpi-common=4.0.3-0ubuntu1 -y && \
    apt-get install --no-install-recommends build-essential=12.8ubuntu1.1 -y && \
    rm -rf /var/lib/apt/lists/*


#Setup SSH for MPI distributed training
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server=1:8.2p1-4ubuntu0.5 && mkdir -p /var/run/sshd && \
    sed 's@session\\s*required\\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
    mkdir -p /root/.ssh/ && ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config && \
    rm -rf /var/lib/apt/lists/*


#Enable Intel optimized Tensorflow
ENV TF_ENABLE_ONEDNN_OPTS=1
ENV HOROVOD_CPU_OPERATIONS=MPI

#Install the environment
RUN pip3 install --no-cache-dir sagemaker-training==4.1.1 && \
    pip3 install --no-cache-dir tensorflow-cpu==2.9.1 && \
    pip3 install --no-cache-dir transformers==4.19.2 && \
    pip3 install --no-cache-dir datasets==2.2.2 && \
    pip3 install --no-cache-dir pandas==1.2.5 && \
    pip3 install --no-cache-dir neural-compressor==1.14.1 && \
    pip3 install --no-cache-dir protobuf==3.19.6 && \
    pip3 install --no-cache-dir horovod==0.24.2 && \
    pip3 install --no-cache-dir mpi4py==3.1.3

#Copy the training files to the container
COPY train.py /opt/ml/code/train.py
COPY ptq_config.yaml /opt/ml/ptq_config.yaml

# # Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py