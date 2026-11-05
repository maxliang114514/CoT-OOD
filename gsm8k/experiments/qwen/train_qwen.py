"""
T-CotMechanism风格训练脚本 - Qwen2.5版本
参考: T-CotMechanism/RealisticDataVerification/train.py

关键特性:
- Qwen2.5 chat format: <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant
- LoRA训练 (rank=16, alpha=16)
"""

import copy
import random
import json
import logging
import os
import sys
import io
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import math

import torch
import torch.nn as nn
import transformers
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)



PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:"
)

IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-3B")


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    max_train_samples: int = field(default=100000)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    merge: bool = field(default=False)
    report_to: str = field(default="none")


def get_model(model_args, training_args, tokenizer):
    """构建模型"""
    logger.info(f"Loading model from {model_args.model_name_or_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )


    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        init_lora_weights=True,
    )
    logger.info(f"Applying LoRA: rank={training_args.lora_rank}, alpha={training_args.lora_alpha}")
    model = get_peft_model(model, lora_config)


    trainable_params, total_params = 0, 0
    for p in model.parameters():
        num_params = p.numel()
        if p.requires_grad:
            trainable_params += num_params
        total_params += num_params
    logger.info(f"Trainable params: {trainable_params:,} / Total params: {total_params:,} = {trainable_params/total_params*100:.2f}%")

    return model


class SupervisedDataset(Dataset):
    """监督微调数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512, max_train_samples: int = 100000):
        super().__init__()


        logger.info(f"Loading data from {data_path}")
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.list_data_dict = json.load(f)
        elif data_path.endswith('.jsonl'):
            self.list_data_dict = []
            with open(data_path, 'r') as f:
                for line in f:
                    self.list_data_dict.append(json.loads(line.strip()))

        logger.info(f"Loaded {len(self.list_data_dict)} samples")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


        random.seed(42)
        random.shuffle(self.list_data_dict)


        if max_train_samples > 0:
            self.list_data_dict = self.list_data_dict[:max_train_samples]
            logger.info(f"Truncated to {len(self.list_data_dict)} samples for training")


        if len(self.list_data_dict) > 0:
            sample = self.list_data_dict[0]
            logger.info("=" * 80)
            logger.info(f"Sample keys: {sample.keys()}")
            if 'query' in sample:
                logger.info(f"Query: {sample['query'][:100]}...")
            if 'response' in sample:
                logger.info(f"Response: {sample['response'][:100]}...")
            if 'type' in sample:
                logger.info(f"Type: {sample['type']}")
            logger.info("=" * 80)

    def __len__(self):
        return len(self.list_data_dict)

    def _tokenize_fn(self, strings, tokenizer) -> Dict:
        """Tokenize字符串列表"""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(input_ids=input_ids, labels=labels, input_ids_lens=input_ids_lens, labels_lens=labels_lens)

    def preprocess(self, sources: Sequence[str], targets: Sequence[str]) -> Dict:
        """预处理数据"""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [self._tokenize_fn(strings, self.tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        item = self.list_data_dict[i]


        if 'query' in item:
            question = item['query']
        elif 'instruction' in item:
            question = item['instruction']
        else:
            question = str(item)

        if 'response' in item:
            answer = item['response']
        elif 'output' in item:
            answer = item['output']
        else:
            answer = ""



        if 'input' in item and item['input']:
            context = item['input'].strip()
            source = PROMPT_TEMPLATE.format(instruction=question) + "\n\n" + context
        else:
            source = PROMPT_TEMPLATE.format(instruction=question)
        target = f"{answer}{self.tokenizer.eos_token}"

        data_dict = self.preprocess([source], [target])
        return {
            'input_ids': data_dict['input_ids'][0],
            'labels': data_dict['labels'][0]
        }


@dataclass
class DataCollatorForSupervisedDataset:
    """数据整理器"""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    logger.setLevel(logging.INFO)

    logger.info("=" * 80)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Data: {data_args.data_path}")
    logger.info(f"Output: {training_args.output_dir}")
    logger.info(f"LoRA: rank={training_args.lora_rank}, alpha={training_args.lora_alpha}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Max length: {training_args.model_max_length}")
    logger.info("=" * 80)


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Tokenizer loaded")


    model = get_model(model_args, training_args, tokenizer)


    train_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        max_train_samples=data_args.max_train_samples
    )


    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )


    trainer.train()


    trainer.save_state()
    if training_args.merge:
        logger.info("Merging LoRA adapter...")
        model = model.merge_and_unload()
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    logger.info(f"Training complete. Output saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()