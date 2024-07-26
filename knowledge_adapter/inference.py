import os
import sys
import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import operator
from operator import itemgetter
from transformers import AdamW, get_linear_schedule_with_warmup
import subprocess
from subprocess import call
import argparse
from dataclass import Data
from evaluation import eval_multi_ref_bleu
from transformers.adapters import AdapterConfig



def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--dataset_prefix', type=str, help="the path that stores the data")
    parser.add_argument('--context_prefix', type=str, help="where the reranked context are.")
    parser.add_argument('--max_table_len', type=int, default=300, help="maximum table length")
    parser.add_argument('--max_tgt_len', type=int, default=90, help="maximum context length")
    parser.add_argument('--max_context_len', default=50, type=int)
    parser.add_argument('--max_content_len', default=5, type=int)
    parser.add_argument('--context_num', type=int, default=3,
                        help="number of prototypes used by the generator")
    parser.add_argument('--add_context', type=str, default='True')
    parser.add_argument('--add_content', type=str, default='True')
    # model configuration
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--max_decode_len', type=int, default=90, help="maximum context length")
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--prefix_adapter_path', type=str)
    return parser.parse_args()


import argparse

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print('Cuda is available.')
    else:
        print('Cuda is not available.')


    train_masked_knowledge_path, train_knowledge_path = args.dataset_prefix + '/train_masked_knowledge.txt', args.dataset_prefix + '/train_knowledge.txt'
    dev_masked_knowledge_path, dev_knowledge_path = args.dataset_prefix + '/dev_masked_knowledge.txt', args.dataset_prefix + '/dev_knowledge.txt'

    train_data_num = 50
    print('Start loading data...')
    data = Data(train_masked_knowledge_path, train_knowledge_path, args.train_data_num,
                dev_masked_knowledge_path, dev_knowledge_path, args.max_table_len,
                args.max_tgt_len, args.model_name)
    print('Data Loaded.')

    print('Loading Model.')

    if args.model_name.startswith('facebook/bart'):
        print('Use Bart Model...')
        from modelling.BARTModel import BARTGen_Model

        model = BARTGen_Model(model_name=args.ckpt_path, tokenizer=data.decode_tokenizer,
                              max_decode_len=args.max_decode_len, dropout=0.0)
    else:
        raise Exception('Wrong Model Mode!!!')



    # Add Bottleneck adapter to the model
    config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16,
                               non_linearity="relu")
    adapter_name = model.load_adapter(adapter_name_or_path=args.knowledge_adapter_path,
                                      model_name="bottleneck_adapter", config=config)

    print("Active adapter: {}".format(adapter_name))
    model.set_active_adapters(adapter_name)

    if torch.cuda.is_available():
        model = model.cuda(device)
    print('Model Loaded.')
    model.eval()

    dev_num = data.dev_num
    batch_size = args.batch_size
    dev_step_num = int(dev_num / batch_size) + 1

    test_output_dir = args.ckpt_path
    dev_ref_text_list, dev_pred_text_list = [], []
    dev_output_text_list, dev_reference_text_list = [], []
    with torch.no_grad():
        # test in-domain evaluation
        print('Perform Evaluation...')
        import progressbar

        p = progressbar.ProgressBar(dev_step_num)
        p.start()
        for dev_step in range(dev_step_num):
            p.update(dev_step)
            dev_batch_src_tensor, dev_batch_src_mask, dev_batch_tgt_in_tensor, dev_batch_tgt_out_tensor, \
            dev_batch_reference_text_list = data.get_next_dev_batch(batch_size)

            if cuda_available:
                dev_batch_src_tensor = dev_batch_src_tensor.cuda(device)
                dev_batch_src_mask = dev_batch_src_mask.cuda(device)
                dev_batch_tgt_in_tensor = dev_batch_tgt_in_tensor.cuda(device)
                dev_batch_tgt_out_tensor = dev_batch_tgt_out_tensor.cuda(device)

            decoded_result = model.generate(dev_batch_src_tensor, dev_batch_src_mask,
                                            tokenized_data=True)
            dev_output_text_list += decoded_result
            dev_reference_text_list += dev_batch_reference_text_list
        p.finish()

        dev_output_text_list = dev_output_text_list[:dev_num]
        dev_reference_text_list = dev_reference_text_list[:dev_num]

        dev_bleu = eval_multi_ref_bleu([dev_reference_text_list], dev_output_text_list,
                                       test_output_dir, 'test_evaluation_')
        print('Test BLEU is %.5f' % dev_bleu)

        output_path = test_output_dir + '/knowledge_test_result_bleu_{}'.format(dev_bleu) + '.txt'
        with open(output_path, 'w', encoding='utf8') as o:
            for text in dev_output_text_list:
                o.writelines(text + '\n')

