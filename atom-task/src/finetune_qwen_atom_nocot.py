#!/usr/bin/env python3
"""
Atom Task (符号替换任务) 微调脚本 - Qwen模型 (No-CoT 版本)
不生成思维链，直接输出最终答案
"""

import os
import json
import argparse
import warnings
import random
import re
from typing import List, Dict, Tuple
import numpy as np
import torch
from tqdm import tqdm

# 设置全局种子，保证实验可复现
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 导入transformers和peft
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
    )
except ImportError as e:
    print(f"Warning: Required library not installed: {e}")
    exit(1)

# ==================== 配置 ====================
# 本文件位于 <项目根目录>/src/，默认 prompt 位于 <项目根目录>/src/prompts/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_SRC_DIR)

# No-CoT Prompt 模板路径
NOCOT_PROMPT_PATH = os.path.join(_SRC_DIR, 'prompts', 'prompt_nocot_atom.txt')

# 默认训练参数
DEFAULT_CONFIG = {
    "output_dir": "./output_qwen_atom_nocot",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "logging_steps": 50,
    "save_steps": 500,
    "save_total_limit": 3,
    "max_seq_length": 1024,
    "gradient_accumulation_steps": 8,
}

EVAL_SAMPLES = 100

# 全局变量
_TOKENIZER_PATH = None
_MAX_LENGTH = None


def load_jsonl(file_path: str) -> list:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: list, file_path: str):
    """保存JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_nocot_template(template_path: str = None) -> str:
    """加载 No-CoT 提示模板"""
    if template_path is None:
        template_path = NOCOT_PROMPT_PATH
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def convert_to_nocot_format(sample: dict) -> str:
    """将 sample 的 output 转换为 No-CoT 格式

    格式: Final Answer: [xxx]
    """
    output = sample.get('output', '')
    return f"Final Answer: [{output}]"


def format_instruction_nocot(sample: dict, tokenizer) -> str:
    """将样本格式化为 Qwen 模型的指令格式 (No-CoT)

    prompt 包含任务定义但不包含示例，要求直接输出最终答案
    """
    prompt_template = load_nocot_template()

    input_text = sample.get('input', '')
    task_chain = sample.get('task_chain_readable', sample.get('task_chain', ''))

    prompt = prompt_template.replace('{input_sequence}', input_text)
    prompt = prompt.replace('{task_chain}', task_chain)

    # 输出格式: Final Answer: [xxx]
    output = sample.get('output', '')
    output_text = f"Final Answer: [{output}]"

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": output_text}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False  # 禁用 thinking 模式
    )

    return text


def preprocess_function(examples, tokenizer, max_length: int = 1024):
    """预处理函数，用于数据集映射

    直接使用原始数据格式进行格式化
    """
    texts = []
    for i in range(len(examples['input'])):
        # 构建样本字典，使用与原始数据一致的字段名
        sample_dict = {
            'input': examples['input'][i],
            'output': examples['output'][i],
            'task_chain_readable': examples.get('task_chain_readable', [''])[i] if 'task_chain_readable' in examples else '',
            'task_chain': examples.get('task_chain', [''])[i] if 'task_chain' in examples else '',
        }
        text = format_instruction_nocot(sample_dict, tokenizer)
        texts.append(text)

    # 逐条tokenize，与原版一致
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for text in texts:
        result = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=True,
            return_tensors=None
        )
        all_input_ids.append(result['input_ids'])
        all_attention_mask.append(result['attention_mask'])
        all_labels.append(result['input_ids'].copy())

    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'labels': all_labels
    }


def load_model(model_path: str):
    """加载模型和分词器"""
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 关键：确保model.config.pad_token_id与tokenizer一致
    if tokenizer.pad_token is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def setup_lora(model):
    """设置 LoRA"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_model(model, tokenizer, train_data, output_dir: str,
                num_train_epochs: int = 3, per_device_train_batch_size: int = 1,
                learning_rate: float = 5e-5, warmup_steps: int = 100,
                logging_steps: int = 50, save_steps: int = 500,
                gradient_accumulation_steps: int = 4, max_grad_norm: float = 1.0):
    """训练模型"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        max_grad_norm=max_grad_norm,
        fp16=True,  # 与原版一致，使用 fp16
        dataloader_num_workers=0,  # 禁用多线程数据加载
        remove_unused_columns=False,
        report_to="none",  # 禁用 wandb
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
    )

    trainer.train()
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Atom Task Fine-tuning (No-CoT)")
    parser.add_argument("--model_path", type=str, default="",
                        help="模型路径（本地 Qwen3-4B 路径，必填）")
    parser.add_argument("--train_file", type=str, required=True,
                        help="训练数据文件 (JSONL)")
    parser.add_argument("--output_dir", type=str, default="./output_qwen_atom_nocot",
                        help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="每设备训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--prompt_file", type=str, default=NOCOT_PROMPT_PATH,
                        help="No-CoT prompt 模板路径")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大训练样本数 (None=全部)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="跳过训练前评估")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()
    set_seed(args.seed)

    global NOCOT_PROMPT_PATH
    NOCOT_PROMPT_PATH = args.prompt_file

    print("=" * 60)
    print("Qwen Atom Task 微调 (No-CoT 版本)")
    print("=" * 60)
    print(f"随机种子: {args.seed}")
    print(f"模型: {args.model_path}")
    print(f"训练数据: {args.train_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.num_train_epochs}")
    print(f"Prompt: {NOCOT_PROMPT_PATH}")
    print("=" * 60)

    # 加载数据
    train_data = load_jsonl(args.train_file)
    if args.max_samples:
        train_data = train_data[:args.max_samples]
    print(f"训练样本数: {len(train_data)}")

    # 转换为数据集格式 - 使用原始数据字段
    train_dataset = Dataset.from_dict({
        'input': [d.get('input', '') for d in train_data],
        'output': [d.get('output', '') for d in train_data],
        'task_chain_readable': [d.get('task_chain_readable', '') for d in train_data],
        'task_chain': [d.get('task_chain', '') for d in train_data],
    })

    # 加载模型
    model, tokenizer = load_model(args.model_path)

    # 设置 LoRA
    model = setup_lora(model)

    # 启用 gradient checkpointing 以减少内存使用
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    model.gradient_checkpointing_enable()

    # 预处理数据
    preprocess_fn = lambda examples: preprocess_function(examples, tokenizer)
    train_dataset = train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="预处理数据"
    )

    # 训练
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_data=train_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
    )

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n最终模型已保存: {final_model_path}")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()