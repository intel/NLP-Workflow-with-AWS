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

#Docker file for Inference
#Note: Users need to specify the region (i.e. us-west-1 and related account) in which they can have access to it.
#Reference: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.9.0-cpu-py39-ubuntu20.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

#Enable Intel optimized Tensorflow
ENV TF_ENABLE_ONEDNN_OPTS=1

#Set the environment variable if the model is quantized (8-bit) by Intel Neural Compressor
ENV TF_ENABLE_MKL_NATIVE_FORMAT=0

#Copy the BERT tokenizer
COPY /bert_uncased_tokenizer /root/assets/sagemaker_inference_container/bert_uncased_tokenizer