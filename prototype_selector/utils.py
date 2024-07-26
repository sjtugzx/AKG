import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import sentence_bleu


def eval_bleu_score(pred_text_list, ref_text_list):
    score_list = []
    assert len(pred_text_list) == len(ref_text_list)
    for idx in range(len(pred_text_list)):
        one_pred_text_list = pred_text_list[idx].split()
        one_ref_text_list = ref_text_list[idx].split()
        one_score = sentence_bleu([one_ref_text_list], one_pred_text_list)
        score_list.append(one_score)
    return np.mean(score_list) * 100 

def extract_test_pred_text(test_batch_score, test_batch_candidate_text_list):
    pred_text_list = []
    test_batch_score_list = test_batch_score.detach().cpu().numpy()
    bsz = len(test_batch_score_list)
    for idx in range(bsz):
        one_score_list = test_batch_score_list[idx]
        one_select_idx = np.argsort(one_score_list)[::-1][0]
        one_select_text = test_batch_candidate_text_list[idx][one_select_idx]
        pred_text_list.append(one_select_text)
    return pred_text_list

def train_parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset_name', type=str, help="human or song or book")
    parser.add_argument('--model_name', type=str, help="e.g. bert-base...")
    parser.add_argument('--dataset_prefix', type=str, help="the path that stores the data")
    parser.add_argument('--train_data_num', type=int)
    parser.add_argument('--train_candi_pool_size', type=int, default=50, 
        help="Randomly selecting negative examples from the top-k retrieved candidates provided by the IR system.")
    parser.add_argument('--train_negative_num', type=int, default=5, 
        help="number of randomly selected negatives from the retrieved candidates from the IR system.")
    parser.add_argument('--test_candi_span', type=int, default=20, 
        help="reranking the best response from top-n candidates from the IR system.")
    parser.add_argument('--max_table_len', type=int, default=250)
    parser.add_argument('--max_tgt_len', type=int, default=80)
    # training configuration
    parser.add_argument('--loss_margin', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--total_steps', type=int)
    parser.add_argument('--update_steps', type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--eval_every', type=int)
    # learning configuration
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_output_dir', type=str)
    return parser.parse_args()