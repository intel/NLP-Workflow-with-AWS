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
import pandas as pd
import sys
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizerFast
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from transformers import TFAutoModelForSequenceClassification
from datasets import load_metric
from transformers import Trainer
from transformers import TrainingArguments
from neural_compressor.experimental import Quantization, common
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver
from tensorflow.python.framework import convert_to_constants
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.eager import context
from neural_compressor.metric import METRICS
from tensorflow import keras

import horovod.tensorflow as hvd
from tqdm import tqdm

# Apply Transfer Learning/Fine-tuning on HuggingFace model
def train_hf_bert(output_model_path):
    def get_datasets():
        def tokenize_function(example):
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", max_length=args.max_seq_length)

        def prepare_tf_dataset(dataset):
            dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "token_type_ids","label"])
            features = {sample: dataset[sample] for sample in ["attention_mask", "input_ids", "token_type_ids"]}
            tf_dataset = tf.data.Dataset.from_tensor_slices((features, dataset["label"]))
            return tf_dataset

        train_dataset, test_dataset = load_dataset('glue', 'mrpc', split=["train", "test"])
        train_dataset, test_dataset = train_dataset.map(tokenize_function, batched=True), test_dataset.map(tokenize_function, batched=True)

        tf_train_dataset, tf_test_dataset = prepare_tf_dataset(train_dataset), prepare_tf_dataset(test_dataset)
        tf_train_dataset_shard,  tf_test_dataset_shard = tf_train_dataset.shard(hvd.size(), hvd.rank()) , tf_test_dataset.shard(hvd.size(), hvd.rank())
        tf_train_dataset_batch , tf_test_dataset_batch = tf_train_dataset_shard.batch(args.train_batch_size, drop_remainder=True) , tf_test_dataset_shard.batch(args.eval_batch_size, drop_remainder=True)

        return tf_train_dataset_batch, tf_test_dataset_batch


    # initialization
    model = TFAutoModelForSequenceClassification.from_pretrained(args.model_name)
    tf_train_dataset, tf_test_dataset = get_datasets()

    # specify the devices to use Intel's CPU
    cpus = tf.config.experimental.list_physical_devices("CPU")
    if cpus:
        tf.config.experimental.set_visible_devices(cpus[hvd.local_rank()], "CPU")

    # fine optimizer and loss
    #decay_steps = int(1374/hvd.size()) #int(len(tf_train_dataset)/args.train_batch_size * args.epochs)
    complete_epoch_steps_mrpc_batchsize_16 = 230 #hyper-parameters tuning
    decay_steps = int((complete_epoch_steps_mrpc_batchsize_16*args.epochs)/hvd.size())

    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(float(args.learning_rate)*hvd.size(),
        decay_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False
    )
 
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        amsgrad=False)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Training
    if args.train:
        for epoch in range(args.epochs):
            print("Training - epoch: " + str(epoch))
            progrss = tqdm(tf_train_dataset)
            for i, batch in enumerate(progrss):
                with tf.GradientTape() as tape:
                    _ , targets = batch
                    probs = model(batch).logits
                    loss_value = loss(targets, probs)

                tape = hvd.DistributedGradientTape(tape)

                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if i == 0: #First batch boardcast
                    hvd.broadcast_variables(model.variables, root_rank=0)
                    hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        if hvd.rank() == 0:
            print("----------------------------------------------")
            print("Loss after training: " + str(loss_value.numpy()))
            print("----------------------------------------------")

    # Evaluation
    if args.eval and (hvd.rank() == 0):
        eval_result = model.evaluate(tf_test_dataset, batch_size=args.eval_batch_size, return_dict=True)
        print("----------------------------------------------")
        print("Evaluation: ")
        for k, value in eval_result.items():
            print(str(k) + ": " + str(value))
        print("----------------------------------------------")
     

    # Save result and perform PTQ on a single node
    if hvd.rank() == 0:
         model.save_pretrained(output_model_path, saved_model=True) # save it with saved_model=True in order to have a SavedModel version along with the h5 weights.

         #Convert the SavedModel to pb
         save_pb_path = '/tmp/ConvertedModel.pb'
         saved_model_path = os.path.join(output_model_path, 'saved_model/1')
         convert_savedmodel(saved_model_path, save_pb_path)
        
         #Use INC to apply quantization on the pb
         ptq_model_path = '/opt/ml/model/ptq_model'
         ptq_config_path = '/opt/ml/ptq_config.yaml'
         ptq_inc(ptq_config_path, save_pb_path, ptq_model_path)
    return
    

class Custom_Dataset_MRPC_Test_v2(object):
    def __init__(self):
        def tokenize_function(example):
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", max_length=args.max_seq_length)

        train_dataset, test_dataset = load_dataset('glue', 'mrpc', split=["train", "test"])
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        test_dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "token_type_ids","label"])
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
        return (attention_mask, inputs_ids, token_type_ids), label

    def __len__(self):
        return len(self.test_features['attention_mask'])


# A function to apply PTQ on the saved pb model using Intel Neural Compressor
def ptq_inc(ptq_config_path, pb_output_path, ptq_model_output_path):
    quantizer = Quantization(ptq_config_path)
    quantizer.model = common.Model(pb_output_path)

    mrpc_test_dataset = Custom_Dataset_MRPC_Test_v2()
    quantizer.calib_dataloader = common.DataLoader(mrpc_test_dataset)
    quantizer.eval_dataloader = common.DataLoader(mrpc_test_dataset)
    q_model = quantizer.fit()
    q_model.save(ptq_model_output_path)
    return
    
# A function to convert the SavedModel to pb format    
def convert_savedmodel(model_path, output_pbfile):
    assert context.executing_eagerly()

    model = tf.keras.models.load_model(model_path)
    model.summary()
    func = model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    frozen_func = convert_to_constants.convert_variables_to_constants_v2(func)
    grappler_meta_graph_def = saver.export_meta_graph(graph_def=frozen_func.graph.as_graph_def(), graph=frozen_func.graph)
    fetch_collection = meta_graph_pb2.CollectionDef()

    for array in frozen_func.inputs + frozen_func.outputs:
        fetch_collection.node_list.value.append(array.name)
        
    grappler_meta_graph_def.collection_def["train_op"].CopyFrom(fetch_collection)
    grappler_session_config = config_pb2.ConfigProto()
    rewrite_options = grappler_session_config.graph_options.rewrite_options
    rewrite_options.min_graph_nodes = -1

    opt = tf_optimizer.OptimizeGraph(grappler_session_config, grappler_meta_graph_def, graph_id=b"tf_graph")
    f = gfile.GFile(output_pbfile, 'wb')
    f.write(opt.SerializeToString())
    print("SavedModel is converted into pb format")
    return 

if __name__ == '__main__':
    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sm-model-dir', type=str, default='/opt/ml/model') #default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=2e-5)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--eval", type=bool, default=True)
    parser.add_argument("--model_dir", type=str, default='/opt/ml/model')# os.environ["SM_MODEL_DIR"])
    
    args, _ = parser.parse_known_args()
    sm_model_dir = args.sm_model_dir
    
    #initialize HVD and loggers
    hvd.init()

    #Perform training
    train_hf_bert(os.path.join(sm_model_dir, 'HF_BertBase_MRPC'))