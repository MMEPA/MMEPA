import torch
from torch import nn
import torch.nn.functional as F
from modules.encoders import *
from transformers import BertTokenizer, BertConfig, BertModel, GPT2Config, GPT2Model, GPT2Tokenizer
from peft import LoraConfig, TaskType, get_peft_model
from modules.xbert import XBertModel
from modules.xgpt2 import XGPT2Model
from modules.transformer import Transformer
bert_path = '/data1/kezhou/pretrained_large_model/bert-base-uncased'
GPT2_path = "/data1/kezhou/LLMs/GPT2/small"
# Bert_path = "/data1/kezhou/pretrained_large_model/bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
GPT2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_path)
GPT2_tokenizer.pad_token = GPT2_tokenizer.eos_token
class MPABertTextEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        config = BertConfig.from_pretrained(bert_path)
        if not hasattr(config,'TopK'):
            setattr(config,'TopK',hp.TopK)
        if not hasattr(config,'rank'):
            setattr(config,'rank',hp.rank)
        if not hasattr(config,'audio_dim'):
            setattr(config,'audio_dim',hp.audio_dim)
        if not hasattr(config,'vision_dim'):
            setattr(config,'vision_dim',hp.vision_dim)
        self.rank = hp.rank
        peft_config = LoraConfig(inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1)
        model = XBertModel.from_pretrained(bert_path, config=config)
        self.model = get_peft_model(model, peft_config)
        for n, p in self.model.named_parameters():
            if 'adapter' in n:
                p.requires_grad=True
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def forward(self, text, vision, audio):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids,
                                            vision=vision,
                                            audio=audio)
        return last_hidden_states
    
class MPAGPT2TextEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        config = GPT2Config.from_pretrained(GPT2_path)
        if not hasattr(config,'TopK'):
            setattr(config,'TopK',hp.TopK)
        if not hasattr(config,'rank'):
            setattr(config,'rank',hp.rank)
        self.rank = hp.rank
        peft_config = LoraConfig(inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1)
        model = XGPT2Model.from_pretrained(GPT2_path, config=config)
        self.model = get_peft_model(model, peft_config)
        for n, p in self.model.named_parameters():
            if 'adapter' in n:
                p.requires_grad=True
    
    def forward(self, text, vision, audio):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask = text[:,:,0].long(), text[:,:,1].float()
        last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            vision=vision,
                                            audio=audio)[0]
        return last_hidden_states
class MPA(nn.Module):
    def __init__(self,hp):
        super(MPA, self).__init__()
        self.model = MPABertTextEncoder(hp)
        self.cls_head = SubNet(in_size=768, hidden_size=128, n_class=1, dropout=0.2)
        # self.pad_token_id = 50256
        
    def forward(self, vision, audio, text):   
        batch_size = text.shape[0]
        input_ids, input_mask = text[:,:,0].long(), text[:,:,1].float()
        hidden_states = self.model(text, vision, audio)
        # sequence_lengths = (torch.eq(input_ids, self.pad_token_id).long().argmax(-1) - 1).to('cuda')
        
        # embedding = hidden_states[torch.arange(batch_size, device='cuda'), sequence_lengths]
        embedding = hidden_states[:,0,:]
        pred = self.cls_head(embedding)
        return pred
    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")

class BertLora(nn.Module):
    def __init__(self, hp):
        super().__init__()
        config = BertConfig.from_pretrained(bert_path)
        peft_config = LoraConfig(inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1)
        model = BertModel.from_pretrained(bert_path, config=config)
        self.model = get_peft_model(model, peft_config)
        self.cls_head = SubNet(in_size=768, hidden_size=128, n_class=1, dropout=0.2)

    def forward(self, text, vision, audio):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]
        x_cls = last_hidden_states[:,0,:]
        pred = self.cls_head(x_cls)
        return pred
    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")


class GPT2Lora(nn.Module):
    def __init__(self, hp):
        super().__init__()
        config = GPT2Config.from_pretrained(GPT2_path)
        peft_config = LoraConfig(inference_mode=False, r=232, lora_alpha=32, lora_dropout=0.1)
        model = GPT2Model.from_pretrained(GPT2_path, config=config)
        self.model = get_peft_model(model, peft_config)
        self.cls_head = SubNet(in_size=768, hidden_size=128, n_class=1, dropout=0.2)
        self.pad_token_id = 50256

    def forward(self,  vision, audio, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        batch_size = text.shape[0]
        input_ids, input_mask = text[:,:,0].long(), text[:,:,1].float()
        hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask)[0]
        sequence_lengths = (torch.eq(input_ids, self.pad_token_id).long().argmax(-1) - 1).to('cuda')
        embedding = hidden_states[torch.arange(batch_size, device='cuda'), sequence_lengths]
        pred = self.cls_head(embedding)
        return pred
    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()
        print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")
