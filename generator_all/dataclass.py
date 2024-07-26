import sys
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import progressbar
# from transformers.models.bart.modeling_bart import shift_tokens_right

SEP, EOS, CONTEXT_START, CONTENT_START = '<sep>', '<eos>', '<context_start>', '<content_start>'
class Data:
    def __init__(self, train_table_path, train_summary_path, train_context_path, train_content_path, train_data_num, test_table_path, test_summary_path, 
        test_context_path, test_content_path, max_table_len, max_tgt_len, max_context_len, context_num, max_content_len, model_name, add_context=True, add_content=True):
        self.add_context = add_context
        self.add_content = add_content
        if self.add_context and self.add_content:
            print ('Evaluating corpus and content prompted model.')
        elif self.add_context and not self.add_content:
            print('Evaluating only corpus prompted model.')
        elif not self.add_context and self.add_content:
            print('Evaluating only content prompted model.')
        else:
            print ('Evaluating vanilla seq2seq model.')

        self.max_table_len, self.max_tgt_len, self.max_context_len, self.context_num = \
        max_table_len, max_tgt_len, max_context_len, context_num
        self.max_content_len = max_content_len
        self.special_token_list = [SEP, EOS, CONTEXT_START, CONTENT_START]

        self.model_name = model_name
        if self.model_name.startswith('t5'):
            from transformers import T5Tokenizer, T5TokenizerFast, T5Config
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
            self.decode_tokenizer = T5Tokenizer.from_pretrained(model_name)

            config = T5Config.from_pretrained(model_name)
            self.bos_token_id = config.decoder_start_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            PAD = '__PAD__'
            print ('original vocabulary Size %d' % len(self.tokenizer))
            self.tokenizer.add_tokens(self.special_token_list + [PAD])
            self.decode_tokenizer.add_tokens(self.special_token_list + [PAD])
            print ('vocabulary size after extension is %d' % len(self.tokenizer))
            self.pad_token_id = self.tokenizer.convert_tokens_to_ids([PAD])[0]
            
        elif self.model_name.startswith('facebook/bart'):
            from transformers import BartTokenizer, BartTokenizerFast, BartConfig
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.bos_token_id = self.tokenizer.bos_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id

            self.decode_tokenizer = BartTokenizer.from_pretrained(model_name)

            print ('original vocabulary Size %d' % len(self.tokenizer))
            self.tokenizer.add_tokens(self.special_token_list)
            self.decode_tokenizer.add_tokens(self.special_token_list)
            print ('vocabulary size after extension is %d' % len(self.tokenizer))
        else:
            raise Exception('Wrong Model Name!!!')

        print ('Start loading training data...')
        train_src_id_list, train_tgt_id_list, train_src_text_list, train_tgt_text_list = \
        self.load_data(train_table_path, train_summary_path, train_context_path, train_content_path)
        self.train_src_id_list = train_src_id_list[:train_data_num]
        self.train_tgt_id_list = train_tgt_id_list[:train_data_num]
        self.train_src_text_list = train_src_text_list[:train_data_num]
        self.train_tgt_text_list = train_tgt_text_list[:train_data_num]

        # print(self.train_src_id_list)

        print ('Start loading test data...')
        self.dev_src_id_list, self.dev_tgt_id_list, self.dev_src_text_list, self.dev_tgt_text_list = \
        self.load_data(test_table_path, test_summary_path, test_context_path, test_content_path)

        self.train_num, self.dev_num = len(self.train_src_id_list), len(self.dev_src_id_list)
        print ('train number is %d, test number is %d' % (self.train_num, self.dev_num))
        self.train_idx_list = [i for i in range(self.train_num)]
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.dev_current_idx = 0 

    def load_one_text_id(self, text, max_len):
        text_id_list = self.tokenizer.encode(text, max_length=512, truncation=True, add_special_tokens=False)[:max_len]
        return text_id_list

    def load_text_id_list(self, text_list, max_len):
        p = progressbar.ProgressBar(len(text_list))
        p.start()
        res_id_list = []
        idx = 0
        for text in text_list:
            p.update(idx + 1)
            one_id_list = self.load_one_text_id(text, max_len)[:max_len]
            res_id_list.append(one_id_list)
            idx += 1
        p.finish()
        return res_id_list

    def load_one_src_text(self, text):
        if self.model_name.startswith('t5'):
            res_str = 'translate Table to Text: '
        elif self.model_name.startswith('facebook/bart'):
            res_str = ''
        else:
            raise Exception('Wrong Model Mode!!!')
        item_list = text.strip('\n').split('\t')
        for item in item_list:
            one_item_list = item.split(':')
            assert len(one_item_list) == 2
            slot_key = one_item_list[0].strip()
            slot_value = one_item_list[1].strip()
            one_res_str = slot_key + ' ' + 'is' + ' ' + slot_value + ' ;'
            res_str += one_res_str + ' '
        return res_str.strip()

    def load_src_text_path(self, path):
        src_text_list = []
        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_src_text = self.load_one_src_text(l.strip('\n'))
                src_text_list.append(one_src_text)
        return src_text_list

    def load_content_id_list(self, content_text_list):
        res_id_list = []
        for text in content_text_list:
            one_res_text = CONTENT_START + ' ' + text
            one_res_id_list = self.load_one_text_id(one_res_text, self.max_content_len)[:self.max_content_len]
            res_id_list += one_res_id_list
        # print(res_id_list, content_text_list)
        return res_id_list
    
    def load_context_id_list(self, context_text_list):
        res_id_list = []
        for text in context_text_list:
            one_res_text = CONTEXT_START + ' ' + text
            one_res_id_list = self.load_one_text_id(one_res_text, self.max_context_len)[:self.max_context_len]
            res_id_list += one_res_id_list
        # print(res_id_list, context_text_list)
        return res_id_list

    def load_data(self, src_path, tgt_path, context_path, content_path):
        table_text_list = self.load_src_text_path(src_path)
        tgt_text_list = []
        with open(tgt_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                tgt_text_list.append(l.strip('\n'))
        assert len(table_text_list) == len(tgt_text_list)
        table_text_id_list = self.load_text_id_list(table_text_list, self.max_table_len)

        # loading context 
        context_text_list = []
        with open(context_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_context_text_list = l.strip('\n').split('\t')[:self.context_num]
                context_text_list.append(one_context_text_list)
        assert len(context_text_list) == len(table_text_id_list)
        context_id_list = []
        for item in context_text_list:
            context_id_list.append(self.load_context_id_list(item))
        
        # loading content
        content_text_list = []
        with open(content_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_content_text_list = l.strip('\n').split('\t')
                content_text_list.append(one_content_text_list)
        assert len(content_text_list) == len(table_text_id_list)
        content_id_list = []
        for item in content_text_list:
            content_id_list.append(self.load_content_id_list(item))

        src_text_id_list = []
        for idx in range(len(table_text_id_list)):
            if self.add_context and self.add_content:
                one_src_text_id_list = table_text_id_list[idx] + content_id_list[idx] + context_id_list[idx]
            elif self.add_context and not self.add_content:
                one_src_text_id_list = table_text_id_list[idx] + context_id_list[idx]
            elif not self.add_context and self.add_content:
                one_src_text_id_list = table_text_id_list[idx] + content_id_list[idx]
            else:
                one_src_text_id_list = table_text_id_list[idx]
            src_text_id_list.append(one_src_text_id_list)

        tgt_text_id_list = self.load_text_id_list(tgt_text_list, self.max_tgt_len)
        tgt_text_id_list = [[self.bos_token_id] + item + [self.eos_token_id] for item in tgt_text_id_list]
        return src_text_id_list, tgt_text_id_list, table_text_list, tgt_text_list

    def shift_tokens_right(self, input_ids, pad_token_id):
        """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
        prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
        prev_output_tokens[:, 1:] = input_ids[:, :-1]
        return prev_output_tokens
    
    def process_decoder_tensor(self, batch_tgt_tensor):
        batch_labels = batch_tgt_tensor
        batch_input = self.shift_tokens_right(batch_labels, self.pad_token_id)
        batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_input, batch_labels 
        # return batch_labels

    def process_t5_decoder_tensor(self, batch_tgt_tensor):
        batch_input = batch_tgt_tensor[:, :-1].clone()
        batch_labels = batch_tgt_tensor[:, 1:].clone()
        batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_input, batch_labels 

    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_src_id_list, batch_tgt_id_list, batch_tgt_text_list = [], [], []
        for idx in batch_idx_list:
            one_src_id_list = self.train_src_id_list[idx]
            batch_src_id_list.append(torch.LongTensor(one_src_id_list))

            one_tgt_id_list = self.train_tgt_id_list[idx]
            batch_tgt_id_list.append(torch.LongTensor(one_tgt_id_list))

            one_tgt_text = self.train_tgt_text_list[idx]
            batch_tgt_text_list.append(one_tgt_text)

        # source-side input
        batch_src_tensor = rnn.pad_sequence(batch_src_id_list, batch_first=True, padding_value=self.pad_token_id)
        # ---- compute src mask ---- #
        batch_src_mask = torch.ones_like(batch_src_tensor)
        batch_src_mask = batch_src_mask.masked_fill(batch_src_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        # target-side input
        batch_tgt_tensor = rnn.pad_sequence(batch_tgt_id_list, batch_first=True, padding_value=self.pad_token_id)
        if self.model_name.startswith('t5'):
            batch_tgt_in_tensor, batch_tgt_out_tensor = self.process_t5_decoder_tensor(batch_tgt_tensor)
        else:
            batch_tgt_in_tensor, batch_tgt_out_tensor = self.process_decoder_tensor(batch_tgt_tensor)
        # loss is not computed for the positions of -100
        return batch_src_tensor, batch_src_mask, batch_tgt_in_tensor, batch_tgt_out_tensor, batch_tgt_text_list

    def get_next_dev_batch(self, batch_size):
        batch_src_id_list, batch_tgt_id_list, batch_tgt_text_list = [], [], []
        if self.dev_current_idx + batch_size < self.dev_num - 1:
            for i in range(batch_size):
                curr_idx = self.dev_current_idx + i

                one_src_id_list = self.dev_src_id_list[curr_idx]
                batch_src_id_list.append(torch.LongTensor(one_src_id_list))

                one_tgt_id_list = self.dev_tgt_id_list[curr_idx]
                batch_tgt_id_list.append(torch.LongTensor(one_tgt_id_list))

                one_tgt_text = self.dev_tgt_text_list[curr_idx]
                batch_tgt_text_list.append(one_tgt_text)
            self.dev_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = self.dev_current_idx + i
                if curr_idx > self.dev_num - 1: 
                    curr_idx = 0
                    self.dev_current_idx = 0
                else:
                    pass
                one_src_id_list = self.dev_src_id_list[curr_idx]
                batch_src_id_list.append(torch.LongTensor(one_src_id_list))

                one_tgt_id_list = self.dev_tgt_id_list[curr_idx]
                batch_tgt_id_list.append(torch.LongTensor(one_tgt_id_list))

                one_tgt_text = self.dev_tgt_text_list[curr_idx]
                batch_tgt_text_list.append(one_tgt_text)
            self.dev_current_idx = 0
        # source-side input
        batch_src_tensor = rnn.pad_sequence(batch_src_id_list, batch_first=True, padding_value=self.pad_token_id)
        # ---- compute src mask ---- #
        batch_src_mask = torch.ones_like(batch_src_tensor)
        batch_src_mask = batch_src_mask.masked_fill(batch_src_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        # target-side input
        batch_tgt_tensor = rnn.pad_sequence(batch_tgt_id_list, batch_first=True, padding_value=self.pad_token_id)
        if self.model_name.startswith('t5'):
            batch_tgt_in_tensor, batch_tgt_out_tensor = self.process_t5_decoder_tensor(batch_tgt_tensor)
        else:
            batch_tgt_in_tensor, batch_tgt_out_tensor = self.process_decoder_tensor(batch_tgt_tensor)
        return batch_src_tensor, batch_src_mask, batch_tgt_in_tensor, batch_tgt_out_tensor, batch_tgt_text_list
