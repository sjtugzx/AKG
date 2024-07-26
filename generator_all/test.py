import sys
import io
import os
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
from dataclass import Data
import numpy as np
import argparse
import logging
import nltk
from transformers import AdamW, get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu

dataset_prefix = '../data/human'
dataset_name = 'human'
model_name = 't5-base'
train_data_num = 2
train_candi_pool_size = 50 # default
train_negative_num = 5 # default
test_candi_span = 20 # default


train_table_path = dataset_prefix + '/train_table.txt'
train_summary_path = dataset_prefix + '/train_reference_summary.txt'
train_context_path = dataset_prefix + '/train_summary_top_100.txt'
train_content_path = dataset_prefix + '/train_content.txt'

dev_table_path = dataset_prefix + '/dev_table.txt'
dev_summary_path = dataset_prefix + '/dev_reference_summary.txt'
dev_context_path = dataset_prefix + '/dev_summary_top_100.txt'
dev_content_path = dataset_prefix + '/dev_content.txt'


model_name = model_name
# data = Data(model_name, train_table_path, train_summary_path, train_context_path, 
#     dev_table_path, dev_summary_path, dev_context_path, train_data_num, 
#     train_candi_pool_size, train_negative_num, test_candi_span, 250, 
#     80)

data = Data(train_table_path, train_summary_path, train_context_path, train_content_path, 
            train_data_num, 
            dev_table_path, dev_summary_path, dev_context_path, dev_content_path, 
            300, 90, 50, 3, 5, model_name, add_context=True, add_content=True)

# print(data.train_table_id_list, data.train_summary_id_list, data.train_summary_text_list)

a, b, c, d, e = data.get_next_train_batch(1)
# print(a, b, c)

print(a[0].size())




