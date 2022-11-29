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

import argparse, os
from urllib.parse import uses_fragment
import pandas as pd
import sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizerFast
from transformers import AutoModelForSequenceClassification
from datasets import load_metric
from neural_compressor.experimental import Quantization, common
from neural_compressor.metric import METRICS
from transformers import Trainer
from transformers import TrainingArguments
from torch.nn import functional as F
from neural_compressor.utils.load_huggingface import save_for_huggingface_upstream

#A custom class for Intel Neural Compressor to perform quantiztion
class Custom_Dataset_MRPC_Test_v2(object):
    def __init__(self):
        def tokenize_function(example):
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", max_length=args.max_seq_length)

        train_dataset, test_dataset = load_dataset('glue', 'mrpc', split=["train", "test"])
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids","label"])
        self.test_features = {
            x: test_dataset[x]
            for x in ["attention_mask", "input_ids", "token_type_ids", "label"]
        }
        return

    def __getitem__(self, index):
        attention_mask = self.test_features['attention_mask'][index]
        inputs_ids = self.test_features['input_ids'][index]
        token_type_ids = self.test_features['token_type_ids'][index]
        label = self.test_features['label'][index]
        return {'input_ids': inputs_ids, 'attention_mask': attention_mask, 'token_type_ids':token_type_ids}, label

    def __len__(self):
        return len(self.test_features['attention_mask'])

#A custom class for Intel Neural Compressor to perform quantiztion
class Custom_Dataset_MRPC_Train_v2(object):
    def __init__(self):
        def tokenize_function(example):
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", max_length=args.max_seq_length)

        train_dataset, test_dataset = load_dataset('glue', 'mrpc', split=["train", "test"])
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids","label"])
        self.train_features = {
            x: train_dataset[x]
            for x in ["attention_mask", "input_ids", "token_type_ids", "label"]
        }
        return

    def __getitem__(self, index):
        attention_mask = self.train_features['attention_mask'][index]
        inputs_ids = self.train_features['input_ids'][index]
        token_type_ids = self.train_features['token_type_ids'][index]
        label = self.train_features['label'][index]
        return {'input_ids': inputs_ids, 'attention_mask': attention_mask, 'token_type_ids':token_type_ids}, label

    def __len__(self):
        return len(self.train_features['attention_mask'])


#A function to define a metric to measure the performance of the model
def compute_metrics_acc(model_output):
    acc_metric = load_metric("accuracy")
    logits, labels = model_output
    return acc_metric.compute(predictions=np.argmax(logits, axis=-1), references=labels)


# Apply quantization on the trained HuggingFace model
def inc_quantization():
    # Retrieve datasets using the HuggingFace model
    def get_datasets():
        def tokenize_function(example):
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", max_length=args.max_seq_length)

        train_dataset, test_dataset = load_dataset('glue', 'mrpc', split=["train", "test"])
        train_dataset, test_dataset = train_dataset.map(tokenize_function, batched=True), test_dataset.map(tokenize_function, batched=True)
        train_dataset, test_dataset = train_dataset.rename_column("label", "labels"), test_dataset.rename_column("label", "labels") #MRPC specific

        train_dataset.set_format(type='torch')
        test_dataset.set_format(type='torch')
        return train_dataset, test_dataset

    # The evaluation function for Intel Neural Compressor 
    def eval_func_for_nc(model_tuned):
        trainer.model = model_tuned
        result = trainer.evaluate(eval_dataset=test_dataset)
        bert_task_acc_keys = ['eval_f1', 'eval_accuracy', 'mcc', 'spearmanr', 'acc']
        for key in bert_task_acc_keys:
            if key in result.keys():
                acc = result[key]
                break
        return acc
    
    # init materials for quantization
    train_dataset, test_dataset = get_datasets()
    model = AutoModelForSequenceClassification.from_pretrained(args.fp32_model_path + '/')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_args = TrainingArguments(
        output_dir="./",
        num_train_epochs=3,
        evaluation_strategy="epoch",
        use_ipex=True,
        bf16=False,
        no_cuda=True
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_acc
    )   

    ptq_config_path = args.inc_config_path
    
    # Start quantization
    quantizer = Quantization(ptq_config_path)
    quantizer.model = common.Model(model)
    mrpc_train_dataset = Custom_Dataset_MRPC_Train_v2()
    quantizer.calib_dataloader = common.DataLoader(mrpc_train_dataset)
    quantizer.eval_func = eval_func_for_nc
    q_model = quantizer.fit()
    #q_model.save(args.quantized_output_path)
    save_for_huggingface_upstream(q_model, tokenizer, args.quantized_output_path)
    print('Quantization Complete!')


    #Testing the accuracy
    model = q_model.model
    print('q_model.model: ')
    print(model)
    correct_pred = 0
    train_dataset, test_dataset = load_dataset('glue', 'mrpc', split=["train", "test"])
    for i in range(len(train_dataset['sentence1'])):
        sentence1 = train_dataset['sentence1'][i]
        sentence2 = train_dataset['sentence2'][i]
        gt = train_dataset['label'][i]
        example_inputs = tokenizer(sentence1, sentence2, return_tensors="pt")

        # Perform inference
        results = model(**example_inputs)
        probability = F.softmax(results['logits'][0], dim=0)
        pred = torch.argmax(probability).item()

        if pred == gt:
            correct_pred = correct_pred + 1

    print('Accuracy: ' + str(correct_pred/len(train_dataset['sentence1'])))
    print('Correct pred: '+  str(correct_pred))
    print("len(train_dataset['sentence1']: " + str(len(train_dataset['sentence1'])))
    return


if __name__ == '__main__':
    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()
    parser.add_argument("--inc_config_path", type=str, default='/opt/ml/code/inc_config.yaml')
    parser.add_argument("--fp32_model_path", type=str, default='/data')
    parser.add_argument("--model_name", type=str, default='bert-base-uncased')
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--quantized_output_path", type=str, default='/data/quantized_model')
    
    args, _ = parser.parse_known_args()

    #Perform quantization
    inc_quantization()
