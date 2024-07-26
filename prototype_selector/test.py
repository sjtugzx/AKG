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
from retriever import trainer

dataset_prefix = '../data/book'
dataset_name = 'book'
model_name = 'bert-base-uncased'
train_data_num = 2
train_candi_pool_size = 50 # default
train_negative_num = 5 # default
test_candi_span = 20 # default


train_table_path = dataset_prefix + '/train_table.txt'
train_summary_path = dataset_prefix + '/train_reference_summary.txt'
train_context_path = dataset_prefix + '/train_summary_top_100.txt'

dev_table_path = dataset_prefix + '/dev_table.txt'
dev_summary_path = dataset_prefix + '/dev_reference_summary.txt'
dev_context_path = dataset_prefix + '/dev_summary_top_100.txt'

assert dataset_name in ["human", "book", "song"]
if dataset_name == 'human':
    remove_slot_key_list = ['caption', 'death date', 'name', 'article title', 
                            'image', 'fullname', 'full name', 'birthname', 'birth name', 'alias', 
                            'othername', 'imdb', '|name', '|imagesize', 'othername',
                            'image caption', 'image size']
elif dataset_name == 'book':
    remove_slot_key_list = ['name', 'author', 'publisher', 'publication date', 'written by', 'country']
else: # song
    remove_slot_key_list = ['name']
remove_key_set = set(remove_slot_key_list)
model_name = model_name
data = Data(model_name, train_table_path, train_summary_path, train_context_path, 
    dev_table_path, dev_summary_path, dev_context_path, train_data_num, 
    train_candi_pool_size, train_negative_num, test_candi_span, 250, 
    80, remove_key_set)

# print(data.train_table_id_list, data.train_summary_id_list, data.train_summary_text_list)

a, b, c = data.get_next_train_batch(1)
# print(a, b, c)

print(a[0].size())




