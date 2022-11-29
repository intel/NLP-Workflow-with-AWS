#!/bin/bash

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

AWS_CLI=${PWD}/aws-cli/bin
REPO_DIR=$1
CONFIG_PATH=${REPO_DIR}/notebooks/config/sagemaker_config.yaml

#Check if aws command exists or not
export PATH=${AWS_CLI}:${PATH}
if ! command -v aws &> /dev/null;
then
    curl --silent "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip";
    unzip -qq awscliv2.zip;
    ./aws/install -i ${PWD}/aws-cli/ -b ${PWD}/aws-cli/bin/ 1> /dev/null;
fi
aws configure import --csv file://${AWS_CSV_FILE}

#Build and push required container
sh ${REPO_DIR}/src/sagemaker_inference_container/build_and_push.sh ${REPO_DIR}/src/sagemaker_inference_container/

#Setup config for notebook
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
echo inference_image_uri: ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/sagemaker-inteltf-huggingface-inc-inference-docker > $CONFIG_PATH
echo quantized_model_s3_path: ${S3_MODEL_URI} >> $CONFIG_PATH
echo role: ${ROLE} >> $CONFIG_PATH