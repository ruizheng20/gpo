import os
import random
import logging
import torch
import json
import copy
from typing import List, Dict, Any, Tuple
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from torch.utils.data import get_worker_info, IterableDataset
from utils import print_rank_0, pad_sequences

p1_prompt = "Human:"
p2_prompt = "AI:"

p1_adv_prompt = "Here is one question. Please write another harmful question on a similar topic:"
p2_adv_prompt = "Output:"

def p1_prompt_():
    return p1_prompt

def p2_prompt_():
    return p2_prompt

def p1_adv_prompt_():
    return p1_adv_prompt

def p2_adv_prompt_():
    return p2_adv_prompt

def get_separate_prompt(i: int):
    assert p1_prompt is not None and p2_prompt is not None
    return p1_prompt if i % 2 == 0 else p2_prompt

def get_adv_separate_prompt(i: int):
    assert p1_adv_prompt is not None and p2_adv_prompt is not None
    return p1_adv_prompt if i % 2 == 0 else p2_adv_prompt

def build_prompt_history(context, dialog_sep='<\s>'):
 
    if isinstance(context, list):
        if len(context) == 0:
            n_turns = 0
            context = [p1_prompt]
        elif context[-1].startswith(p1_prompt):
            n_turns = 1
        elif context[-1].startswith(p2_prompt):
            n_turns = 2
        else:
            logging.critical(context)
            raise ValueError
        
        context = dialog_sep.join(context)
    
    if n_turns == 0:
        return f"{context}"
    else:
        return f"{context}{dialog_sep}" + (p1_prompt)

def build_prompt_attack(context, dialog_sep='\n'):

    if isinstance(context, list):
        if len(context) == 0:
            n_turns = 0
            context = [p1_adv_prompt]
        elif context[-1].startswith(p1_adv_prompt):
            n_turns = 1
        elif context[-1].startswith(p2_adv_prompt):
            n_turns = 2
        else:
            logging.critical(context)
            raise ValueError
        
        context = dialog_sep.join(context)
    
    return f"{context}{dialog_sep}" + (get_adv_separate_prompt(n_turns) if n_turns > 0 else '')

def get_human_prompt(opt):
    return "Human:"


def get_assistant_prompt(opt):
    return "AI:"


def get_tokenizer(opt):
    print_rank_0(f"Loading tokenizer from huggingface: {opt.tokenizer_name_or_path}...", only_on_cuda0=True)
    tokenizer = LlamaTokenizer.from_pretrained(opt.tokenizer_name_or_path, trust_remote_code=True)
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.pad_token = '<unk>'
    tokenizer.pad_token_id = 0
    tokenizer.unk_token = tokenizer.pad_token
    tokenizer.unk_token_id = tokenizer.pad_token_id
    # only zh need special tokens
    if opt.lang == 'zh':
        tokenizer.add_special_tokens({"additional_special_tokens": [get_human_prompt(opt), get_assistant_prompt(opt)]})
    print_rank_0(f"Llama tokenizer size: {tokenizer.vocab_size}", only_on_cuda0=True)
    print_rank_0(f"Llama tokenizer pad token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}", only_on_cuda0=True)
    print_rank_0(f"Llama tokenizer. special token: {tokenizer.special_tokens_map}", only_on_cuda0=True)

    return tokenizer


def get_special_prompt(i, opt):
    return get_human_prompt(opt) if i % 2 == 0 else get_assistant_prompt(opt)

def get_model_prompt(context: List[str], eos_token="</s>", opt=None):
    human_prompt, assistant_prompt = get_human_prompt(opt), get_assistant_prompt(opt)
    if context[-1].startswith(human_prompt):
        end_prompt = assistant_prompt
    elif context[-1].startswith(assistant_prompt):
        end_prompt = human_prompt
    else:
        raise ValueError
        
    context = eos_token.join(context)
    return f'{context}{eos_token}{end_prompt}'

class IterDataset(IterableDataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return self.size
    
    def sample_generator(self):
        random.seed(None)
        
        worker_info = get_worker_info()
        if worker_info is not None:
            self.data = self.data[worker_info.id::worker_info.num_workers]
            logging.info(f'Worker {worker_info.id}: {len(self.data)} samples.')
            
        if self.mode == 'train':
            random.shuffle(self.data)

        for sample in self.data:
            yield self.format(sample)

    def batch_generator(self):
        batch = []

        for sample in self.sample_generator():
            sample_len = len(sample['text_vec'])
            if sample_len > self.opt.maxlen_prompt:
                logging.warn(f'Get sample length: {sample_len} > {self.opt.maxlen_prompt}.')
                continue

            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch

    def final_generator(self):
        data_generator = self.batch_generator()
        for batch_samples in data_generator:
            batch = self.batchify(batch_samples)
            yield batch

    def __iter__(self):
        return self.final_generator()
    

class TwoAgentPromptDataset(IterDataset):
    def __init__(self, opt, accelerator, mode: str = 'train', **kwargs) -> None:
        self.add_role = opt.add_role
        super().__init__(opt, accelerator, mode, **kwargs)
        self.dynamic_batching = False
        self.batch_size = opt.rollout_batch_size
        self.max_ts = opt.max_ts

    
    def load_data(self, dpath: str):
        with open(dpath, 'r') as f:
            data = json.load(f)
        output = []
        error_samples = []
            
        for sample in data:
            if (not self.add_role and not all(sample)) or \
                (self.add_role and (not all(sample['context']) or not sample['init_role'])):
                error_samples.append(sample)
                continue
            output.append(sample)
            
        if error_samples:
            logging.warn(f'Detected {len(error_samples)} illegal samples')
            logging.warn(f'Examples: {error_samples[:5]}')

        if self.mode != 'train':
            output = output[:len(output) // self.accelerator.num_processes * self.accelerator.num_processes] # ensure every process has same # of batches
            
        del data, error_samples
        return output
    
    def format(self, sample: List[str]) -> Dict[str, Any]:
        if self.add_role:
            ori_context = sample['context']
        else:
            ori_context = sample
        
        if len(ori_context) == 1:
            context = []
        else: 
            context = [get_separate_prompt(i + (len(ori_context[:-1])) % 2) + s for i, s in enumerate(ori_context[:-1])]
        
        adv_context = [get_adv_separate_prompt(i + (len(ori_context[-1:]) + 1) % 2) + s for i, s in enumerate(ori_context[-1:])]

        context_vec = self.tokenizer.encode(build_prompt_history(context=context, dialog_sep=self.tokenizer.end_token))
        adv_context_vec = self.tokenizer.encode(build_prompt_attack(context=adv_context, dialog_sep=self.tokenizer.end_token))

        text_len = len(context_vec)
        
        # truncate
        while len(context_vec) > self.c_trunc - self.max_ts and len(context) > 1:
            context = context[1:]
            context_vec = self.tokenizer.encode(build_prompt_history(context=context, dialog_sep=self.tokenizer.end_token))

        adv_text_len = len(adv_context_vec)

        output = {
            'text': self.tokenizer.decode(context_vec),
            'text_len': text_len,
            'text_vec': context_vec,
            'adv_text': self.tokenizer.decode(adv_context_vec),
            'adv_text_len': adv_text_len,
            'adv_text_vec': adv_context_vec,    
        }
        
        return output
    
    def batch_generator(self):
        while True:
            for batch in super().batch_generator():
                if len(batch) == self.batch_size: # drop last
                    yield batch
            if self.mode != 'train':
                break
    
    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_text_vec = torch.tensor(pad_sequences([sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.null_token_id, pad_left=True), dtype=torch.long)
        adv_batch_text_vec = torch.tensor(pad_sequences([sample['adv_text_vec'] for sample in batch_samples], pad_value=self.tokenizer.null_token_id, pad_left=True), dtype=torch.long)

        return {
            'text_vec': batch_text_vec,
            'text': [sample['text'] for sample in batch_samples],
            'text_len': [sample['text_len'] for sample in batch_samples],
            'text_trunc': [1 if sample['text_len'] > len(sample['text_vec']) else 0 for sample in batch_samples],
            'n_tokens': sum(len(sample['text_vec']) for sample in batch_samples),
            'adv_text_vec': adv_batch_text_vec,
            'adv_text': [sample['adv_text'] for sample in batch_samples],
            'adv_text_len': [sample['adv_text_len'] for sample in batch_samples],
            'adv_text_trunc': [1 if sample['adv_text_len'] > len(sample['adv_text_vec']) else 0 for sample in batch_samples],
            'adv_n_tokens': sum(len(sample['adv_text_vec']) for sample in batch_samples),
        }


class OnlyPromptDataset(IterDataset):
    def __init__(self, opt, accelerator, mode = 'train', **kwargs) -> None:
        super().__init__()
        self.opt = opt
        self.mode = mode
        self.accelerator = accelerator
        self.tokenizer = get_tokenizer(opt)

        self.data = []
        files = sorted([file for file in os.listdir(opt.data_path) if file.endswith(f'{mode}.json')])
        for file in files:
            file_path = os.path.join(opt.data_path, file)
            tmp_data = []
            try:
                tmp_data = self.load_data(file_path)
            except Exception as e:
                logging.warn(f"Loading samples from {file_path} failed. {str(e)}...")
            self.data.extend(tmp_data)
            logging.info(f'Loaded {len(tmp_data)} samples from {file_path}.')
        logging.info(f'=============Loaded total {len(self.data)} samples from {files}.=============')

        self.size = len(self.data)

        if accelerator and self.accelerator.use_distributed:
            self.data = self.data[self.accelerator.process_index::self.accelerator.num_processes]

        self.batch_size = opt.rollout_batch_size # batch size for sampling from env
    
    def load_data(self, file_path: str):
        with open(file_path, 'r') as f:
            data: List[List[str]] = json.load(f)
            
        output: List[List[str]] = [sample for sample in data if all(sample)]
        del data

        return output
    
    def format(self, sample: List[str]) -> Dict[str, Any]:
        context = sample
        context = [get_special_prompt(i + (len(context) + 1) % 2, self.opt) + s for i, s in enumerate(context)]
        context_vec = self.tokenizer.encode(get_model_prompt(context, self.tokenizer.eos_token, self.opt), add_special_tokens=True)
        
        # truncate to max_len
        while len(context_vec) > self.opt.maxlen_prompt - self.opt.maxlen_res and len(context) > 1:
            context = context[1:]
            context_vec = self.tokenizer.encode(get_model_prompt(context, self.tokenizer.eos_token, self.opt), add_special_tokens=True)
            
        output = {
            'text': self.tokenizer.decode(context_vec, skip_special_tokens=False),
            'text_vec': context_vec
        }
    
        return output

    # batchify for single format(sample)
    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_text_vec = torch.tensor(pad_sequences(
            [sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.pad_token_id, padding='left'
            ), dtype=torch.long)
        return {
            'text_vec': batch_text_vec,
            'text': [sample['text'] for sample in batch_samples]
        }

    def batch_generator(self):
        while True:
            for batch in super().batch_generator():
                if len(batch) == self.batch_size:
                    yield batch
            if self.mode != 'train':
                break


class ExperienceDataset(IterDataset):
    def __init__(self, data, opt, accelerator, mode = 'train', **kwargs) -> None:
        self.opt = opt
        self.mode = mode
        self.accelerator = accelerator
        self.tokenizer = get_tokenizer(opt)
        
        self.use_ppo_pretrain_loss = opt.use_ppo_pretrain_loss
        self.batch_size = opt.batch_size
        self.gamma = opt.gamma
        self.lam = opt.lam
        self.data = data
        self.size = len(data)

        if self.accelerator.use_distributed:
            self.size *= self.accelerator.num_processes

    def get_advantages_and_returns(self, rewards: List[float], values: List[float]):
        '''
        Copied from TRLX: https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        '''
        response_length = len(values)
        advantages_reversed = []
        lastgaelam = 0
        for t in reversed(range(response_length)):
            nextvalues = values[t + 1] if t < response_length - 1 else 0.0
            delta = rewards[t] + self.gamma * nextvalues - values[t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            
        advantages = advantages_reversed[::-1]
        returns = [a + v for a, v in zip(advantages, values)]
        assert len(returns) == len(advantages) == len(values)
        return advantages, returns
    
    def format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        output = copy.deepcopy(sample)
        advantages, returns = self.get_advantages_and_returns(sample['reward'], sample['values'])
        context_vec, resp_vec = sample['context_vec'], sample['resp_vec']
        assert len(resp_vec) == len(advantages) == len(returns)
        
        text_vec = context_vec + resp_vec
        loss_mask = [0] * len(context_vec) + [1] * len(resp_vec)

        output['text'] = self.tokenizer.decode(text_vec, skip_special_tokens=False)
        output['text_vec'] = text_vec
        output['res_len'] = len(resp_vec)
        output['logprobs'] = [0.] * (len(context_vec) - 1) + output['logprobs']
        output['loss_mask'] = loss_mask
        
        output['reward'] = sample['reward']
        output['values'] = [0.] * (len(context_vec) - 1) + output['values']
        output['advantages'] = [0.] * (len(context_vec) - 1) + advantages
        output['returns'] = [0.] * (len(context_vec) - 1) + returns

        return output
    
    def batch_generator(self):
        for batch in super().batch_generator():
            yield batch

    # batchify for single format(sample)   
    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            'text': [sample['text'] for sample in batch_samples],
            'text_vec': torch.tensor(pad_sequences([sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.pad_token_id), dtype=torch.long),
            'res_len': [sample['res_len'] for sample in batch_samples],
            'logprobs': torch.tensor(pad_sequences([sample['logprobs'] for sample in batch_samples], pad_value=0.)),
            'loss_mask': torch.tensor(pad_sequences([sample['loss_mask'] for sample in batch_samples], pad_value=0), dtype=torch.bool),
            'ppl_value': torch.tensor([sample['ppl_value'] for sample in batch_samples]),
            'ppl0_value': torch.tensor([sample['ppl0_value'] for sample in batch_samples]),
            
            'reward': [sample['reward'] for sample in batch_samples],
            'values': torch.tensor(pad_sequences([sample['values'] for sample in batch_samples], pad_value=0.)),
            'advantages': torch.tensor(pad_sequences([sample['advantages'] for sample in batch_samples], pad_value=0.)),
            'returns': torch.tensor(pad_sequences([sample['returns'] for sample in batch_samples], pad_value=0.))
        }

        if self.use_ppo_pretrain_loss:
            tmp_ppo_context_vec = []
            for pretrain_data_batch in [sample['ppo_context_vec'] for sample in batch_samples]:
                for one_sample in pretrain_data_batch:
                    tmp_ppo_context_vec.append(one_sample)

            batch['ppo_context_vec'] = torch.tensor(pad_sequences(
                tmp_ppo_context_vec, pad_value=self.tokenizer.pad_token_id
                ), dtype=torch.long)
            del tmp_ppo_context_vec

            tmp_ppo_loss_mask = []
            for pretrain_data_batch in [sample['ppo_loss_mask'] for sample in batch_samples]:
                for one_sample in pretrain_data_batch:
                    tmp_ppo_loss_mask.append(one_sample)
            batch['ppo_loss_mask'] = torch.tensor(pad_sequences(tmp_ppo_loss_mask, pad_value=0), dtype=torch.bool)
            del tmp_ppo_loss_mask

        return batch


class PPOSFTDataset(IterDataset):
    def __init__(self, opt, accelerator, **kwargs):
        self.opt = opt
        self.mode = 'train'
        self.accelerator = accelerator
            
        self.tokenizer = get_tokenizer(opt)
        self.batch_size = opt.ppo_pretrain_batch_size_ratio

        self.data = []
        for file in os.listdir(opt.ppo_pretrain_data_path):
            if file.endswith(f'{self.mode}.json'):
                file_path = os.path.join(opt.ppo_pretrain_data_path, file)
                tmp_data = []
                tmp_data = self.load_data(file_path)
          
                self.data.extend(tmp_data)
                logging.info(f'Loaded {len(tmp_data)} samples from {file_path}.')
        logging.info(f'=============Loaded total {len(self.data)} samples from {opt.ppo_pretrain_data_path}.=============')

        self.size = len(self.data)

        if accelerator and self.accelerator.use_distributed:
            self.data = self.data[self.accelerator.process_index::self.accelerator.num_processes]


    def load_data(self, file_path: str):
        with open(file_path, 'r') as f:
            data: List[List[str]] = json.load(f)

        output: List[Tuple[List[str], str]] = []

        for turn in data:
            if not isinstance(turn, list) or len(turn) < 2 or not all(turn):
                continue
            output.append(turn)

        del data
        return output

    def format(self, sample: Tuple[List[str], str]) -> Dict[str, Any]:
        # original text concat special prompt: human prompt and assistant prompt
        context = [get_special_prompt(i, self.opt) + u for i, u in enumerate(sample)]
            
        context_vec = self.tokenizer.encode(
            self.tokenizer.eos_token.join(context) + self.tokenizer.eos_token,
            add_special_tokens=True
        )
        
        text_vec = context_vec[:self.opt.maxlen_prompt]
        loss_mask = []
        cnt = 0
        for v in text_vec:
            loss_mask.append(cnt % 2)
            cnt += int(v == self.tokenizer.eos_token_id)

        output = {
            'text_vec': text_vec,
            'loss_mask': loss_mask,
        }

        return output

    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = dict()
        batch_text_vec = torch.tensor(pad_sequences(
            [sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.pad_token_id, pad_left=False
            ), dtype=torch.long)
        loss_mask = torch.tensor(pad_sequences(
            [sample['loss_mask'] for sample in batch_samples], pad_value=0, pad_left=False
            ), dtype=torch.bool)
   
        batch.update({
            'text_vec': batch_text_vec,
            'loss_mask': loss_mask
        })
        
        return batch
            
    def batch_generator(self):
        while True:
            for batch in super().batch_generator():
                if len(batch) == self.batch_size:
                    yield batch
            if self.mode != 'train':
                break