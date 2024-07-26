import torch
from torch import nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers.adapters import BartAdapterModel
def get_special_token_list(tokenizer):
    res_list = []
    for key in tokenizer.special_tokens_map:
        one_token = tokenizer.special_tokens_map[key]
        if type(one_token) == str:
            res_list.append(one_token)
    return res_list

class BARTGen_Model(nn.Module):
    def __init__(self, model_name, tokenizer, max_decode_len, dropout):
        super().__init__()
        self.tokenizer = tokenizer # tokenizer with extended vocabulary
        self.max_decode_len = max_decode_len
        self.tokenizer_special_token_set = set(get_special_token_list(self.tokenizer))

        print ('Initializing Huggingface BART model...')
        bart_config = BartConfig.from_pretrained(model_name)
        bart_config.__dict__["dropout"] = dropout
		#model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
        self.model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
        print ('Resizing Token Embeddings...')
        self.model.resize_token_embeddings(len(self.tokenizer)) 

    def forward(self, src_input, src_mask, tgt_input, tgt_output):
        src_mask = src_mask.type(src_input.type())
        outputs = self.model(input_ids=src_input, attention_mask=src_mask, decoder_input_ids=tgt_input, labels=tgt_output)
        loss = outputs[0]#.mean()
        return loss

    def generate(self, src_input, src_mask, tokenized_data=False):
        result_list = []
        outputs = self.model.generate(input_ids=src_input, attention_mask=src_mask, max_length=self.max_decode_len)
        for predicted_ids in outputs:
            if tokenized_data:
                one_token_list = self.tokenizer.convert_ids_to_tokens(predicted_ids)
                pred_tokens = []
                for token in one_token_list:
                    if token in self.tokenizer_special_token_set:
                        continue
                    else:
                        pred_tokens.append(token)
                one_result = self.tokenizer.convert_tokens_to_string(pred_tokens)
            else:
                one_result = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
            result_list.append(one_result)
        return result_list

    def save_model(self, ckpt_save_path):
        import os
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        self.model.save_pretrained(ckpt_save_path)
        self.tokenizer.save_pretrained(ckpt_save_path)

    def add_adapter(self, adapter_name: str, config=None, overwrite_ok: bool = False, set_active: bool = False):
        self.model.add_adapter(adapter_name, config, overwrite_ok, set_active)

    def add_adapter_fusion(self, adapter_names, config=None, overwrite_ok: bool = False,
                           set_active: bool = False):
        self.model.add_adapter_fusion(adapter_names, config, overwrite_ok, set_active)

    def apply_to_adapter_layers(self, fn):
        self.model.apply_to_adapter_layers(fn)

    def delete_adapter(self, adapter_name: str):
        self.model.delete_adapter(adapter_name)

    def delete_adapter_fusion(self, adapter_names):
        self.model.delete_adapter_fusion(adapter_names)

    def eject_prefix_tuning(self, name: str):
        self.model.eject_prefix_tuning(name)

    def add_seq2seq_lm_head(self, head_name, overwrite_ok=False):
        self.model.add_seq2seq_lm_head(head_name, overwrite_ok)

    def train_adapter(self, adapter_setup, train_embeddings=False):
        self.model.train_adapter(adapter_setup, train_embeddings)

    def train_adapter_fusion(self, adapter_setup, unfreeze_adapters = False):
        self.model.train_adapter_fusion(adapter_setup, unfreeze_adapters)

    def train_fusion(self, adapter_setup, unfreeze_adapters=False):
        self.model.train_fusion(adapter_setup, unfreeze_adapters)

    def set_active_adapters(self, adapter_setup):
        self.model.set_active_adapters(adapter_setup)

    def save_all_adapters(self, save_directory: str, with_head: bool = True, meta_dict: dict =
    None, custom_weights_loaders = None):
        self.model.save_all_adapters(save_directory, with_head, meta_dict, custom_weights_loaders)

    def save_all_adapter_fusions(self, save_directory: str, meta_dict: dict = None,
                                 custom_weights_loaders=None):
        self.model.save_all_adapter_fusions(save_directory, meta_dict, custom_weights_loaders)

    def save_adapter_fusion(self, save_directory: str, adapter_names, meta_dict: dict = None,
                            custom_weights_loaders=None, with_head=False):
        self.model.save_adapter_fusion(save_directory, adapter_names, meta_dict, custom_weights_loaders, with_head)

    def save_adapter(self, save_directory: str, adapter_name: str, with_head: bool = True,
                     meta_dict: dict = None, custom_weights_loaders=None):
        self.model.save_adapter(save_directory, adapter_name, with_head, meta_dict,
                                custom_weights_loaders)

    def load_adapter(self, adapter_name_or_path: str, config=None, version:str = None, model_name: str = None):
        return self.model.load_adapter(adapter_name_or_path, config, version, model_name)