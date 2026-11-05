#!/usr/bin/env python3
"""
Atom-Task 标准 CoT 微调脚本（Qwen + LoRA）。

训练格式（标准 CoT，每一步记录完整 Task_Chain / Input_Sequence）：
    Step 1: Task_Chain: [Reverse, Duplicate, Rotate] | Input_Sequence: [δωωωδγ] | → Output_Sequence: [γδωωωδ] | Task_Chain: [Duplicate, Rotate]
    Step 2: ...
    Final Answer: [xxx]

> 说明：该文件为公开发布版的重构实现，基于 ablation 版本的微调脚本，
> 将 CoT 格式替换为论文中的标准格式。若您手中有原始版本，可直接替换。
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

# 导入 transformers 和 peft
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

# CoT Prompt 模板路径（标准格式，含 Example）
COT_PROMPT_PATH = os.path.join(_SRC_DIR, 'prompts', 'prompt_cot_atom.txt')

# 默认训练参数
DEFAULT_CONFIG = {
    "output_dir": "./output_qwen_atom",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "logging_steps": 50,
    "save_steps": 500,
    "save_total_limit": 3,
    "max_seq_length": 1024,
    "gradient_accumulation_steps": 4,
}

EVAL_SAMPLES = 100

# 全局变量
_TOKENIZER_PATH = None
_MAX_LENGTH = None


def load_jsonl(file_path: str) -> list:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: list, file_path: str):
    """保存 JSONL 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_cot_template(template_path: str = None) -> str:
    """加载 CoT 提示模板"""
    if template_path is None:
        template_path = COT_PROMPT_PATH
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def convert_to_cot_format(sample: dict) -> str:
    """将 sample 的 output 转换为标准 CoT 格式。

    Step 1: Task_Chain: [Reverse, Duplicate, Rotate] | Input_Sequence: [xxx] | → Output_Sequence: [xxx] | Task_Chain: [Duplicate, Rotate]
    ...
    Final Answer: [xxx]
    """
    output = sample.get('output', '')
    task_chain_str = sample.get('task_chain_readable', sample.get('task_chain', ''))
    reasoning_steps = sample.get('reasoning_steps', [])

    if '→' in task_chain_str:
        task_chain = [t.strip() for t in task_chain_str.split('→')]
    elif '->' in task_chain_str:
        task_chain = [t.strip() for t in task_chain_str.split('->')]
    else:
        task_chain = [task_chain_str.strip()]

    steps = []
    for i, step_info in enumerate(reasoning_steps):
        step_input = step_info.get('input', '')
        step_output = step_info.get('output', '')
        remaining_tasks = task_chain[i + 1:]

        current_task_chain_display = ', '.join(task_chain[i:])
        remaining_task_chain_display = ', '.join(remaining_tasks)

        step_text = (f"Step {i+1}: Task_Chain: [{current_task_chain_display}] "
                     f"| Input_Sequence: [{step_input}] "
                     f"| → Output_Sequence: [{step_output}] "
                     f"| Task_Chain: [{remaining_task_chain_display}]")
        steps.append(step_text)

    return '\n'.join(steps) + f"\nFinal Answer: [{output}]"


def format_instruction_qwen(sample: dict, tokenizer,
                            use_cot_template: bool = True,
                            train_mode: str = "normal",
                            partial_loss: bool = False) -> str:
    """将样本格式化为 Qwen 模型的指令格式。"""
    # 如果数据已有完整的 instruction 字段，直接使用（LlamaFactory 格式）
    if 'instruction' in sample and sample.get('input', '') == '':
        prompt = sample['instruction']
    else:
        if use_cot_template:
            prompt_template = load_cot_template()
        else:
            prompt_template = """You are given a sequence of operations to perform on a symbol list.
Operations: Reverse, Substitute, Duplicate, Deduplicate, Rotate.

Input Sequence: {input_sequence}
Task Chain: {task_chain}

Provide the final answer:"""

        input_text = sample.get('input', '')
        task_chain = sample.get('task_chain_readable', sample.get('task_chain', ''))

        prompt = prompt_template.replace('{input_sequence}', input_text)
        prompt = prompt.replace('{task_chain}', task_chain)
        prompt = prompt.replace('{input}', input_text)

    output = sample.get('output', '')
    if train_mode == "cot_format":
        # 如果 output 已经是 CoT 格式，直接使用
        if 'Step' not in output:
            output = convert_to_cot_format(sample)

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": output}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    return text


def tokenize_function(examples):
    """使用全局变量进行 tokenize"""
    global _TOKENIZER_PATH, _MAX_LENGTH
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    result = tokenizer(
        examples['text'],
        truncation=True,
        max_length=_MAX_LENGTH,
        padding=False,
        return_tensors=None,
        add_special_tokens=True
    )
    return result


def prepare_dataset(train_file: str, tokenizer, max_length: int = 2048,
                    use_cot_template: bool = True,
                    train_mode: str = "normal",
                    partial_loss: bool = False) -> Dataset:
    print(f"加载训练数据: {train_file}")
    data = load_jsonl(train_file)
    print(f"  总样本数: {len(data)}")
    print(f"  训练模式: {train_mode}")

    texts = []
    for i, sample in enumerate(data):
        try:
            text = format_instruction_qwen(sample, tokenizer, use_cot_template, train_mode, partial_loss)
            texts.append(text)
        except Exception as e:
            print(f"  警告: 第{i}条样本格式化失败: {e}")
            continue

    print(f"  成功格式化: {len(texts)} 条")

    global _TOKENIZER_PATH, _MAX_LENGTH
    _TOKENIZER_PATH = tokenizer.name_or_path
    _MAX_LENGTH = max_length

    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  处理数据集...")
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
        tokens = result['input_ids']
        labels = list(tokens)
        all_input_ids.append(tokens)
        all_attention_mask.append(result['attention_mask'])
        all_labels.append(labels)

    return Dataset.from_dict({
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'labels': all_labels
    })


def setup_model(model_path: str, output_dir: str, config: dict):
    print(f"加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def extract_final_answer(text: str, train_mode: str = "normal") -> str:
    """从模型输出中提取最终答案。"""
    text = text.strip()
    text = re.sub(r'<BOS>|<EOS>|<|endoftext|>', '', text)

    if train_mode == "cot_format":
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].strip()
            answer = answer.strip('"').strip("'")
            answer = answer.split('\n')[0].strip()
            if answer.startswith('[') and answer.endswith(']'):
                answer = answer[1:-1]
            return answer

    return text.strip()


def evaluate_on_train(model, tokenizer, train_data: List[Dict],
                      train_mode: str = "normal",
                      use_cot_template: bool = True,
                      max_new_tokens: int = 1024) -> Tuple[List[str], float]:
    """在训练集上评估模型。"""
    predictions = []
    correct = 0

    for sample in train_data:
        if 'instruction' in sample and sample.get('input', '') == '':
            prompt = sample['instruction']
        else:
            if use_cot_template:
                prompt_template = load_cot_template()
            else:
                prompt_template = """You are given a sequence of operations to perform on a symbol list.

Input Sequence: {input_sequence}
Task Chain: {task_chain}

Provide the final answer:"""

            input_text = sample.get('input', '')
            task_chain = sample.get('task_chain_readable', sample.get('task_chain', ''))
            prompt = prompt_template.replace('{input_sequence}', input_text)
            prompt = prompt.replace('{task_chain}', task_chain)
            prompt = prompt.replace('{input}', input_text)
            prompt = prompt.strip()

        target_output = sample['output']

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            content = tokenizer.decode(output_ids, skip_special_tokens=True)

        predictions.append(content)
        predicted_answer = extract_final_answer(content, train_mode)
        target_answer = extract_final_answer(target_output, train_mode)

        if predicted_answer.strip() == target_answer.strip():
            correct += 1

    accuracy = correct / len(train_data) if train_data else 0
    return predictions, accuracy


def run_train_eval(model, tokenizer, train_data: List[Dict], eval_samples: int,
                   task_desc: str, output_dir: str, train_mode: str = "normal",
                   use_cot_template: bool = True) -> Dict:
    """在训练集上运行评估。"""
    if len(train_data) > eval_samples:
        eval_data = random.sample(train_data, eval_samples)
    else:
        eval_data = train_data

    print(f"\n{task_desc}: 测试 {len(eval_data)} 个样本")
    eval_file = os.path.join(output_dir, f"eval_samples_{task_desc.replace(' ', '_')}.jsonl")
    save_jsonl(eval_data, eval_file)

    predictions, accuracy = evaluate_on_train(
        model, tokenizer, eval_data, train_mode, use_cot_template
    )

    print(f"\n{task_desc} 结果: 准确率 {accuracy*100:.2f}%")
    return {
        'task': task_desc,
        'num_samples': len(eval_data),
        'accuracy': accuracy,
        'predictions': predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen Atom Task 微调 - 标准 CoT")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--train_file", type=str, required=True, help="训练数据")
    parser.add_argument("--output_dir", type=str, default="./output_qwen_atom", help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="batch大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--train_mode", type=str, default="normal",
                        choices=["normal", "cot_format"], help="训练模式")
    parser.add_argument("--prompt_file", type=str, default=COT_PROMPT_PATH,
                        help="CoT prompt 模板路径")
    parser.add_argument("--eval_samples", type=int, default=EVAL_SAMPLES, help="评估样本数")
    parser.add_argument("--skip_eval", action="store_true", help="跳过训练前后评估")

    args = parser.parse_args()
    set_seed(args.seed)

    global COT_PROMPT_PATH
    COT_PROMPT_PATH = args.prompt_file

    print("=" * 60)
    print("Qwen Atom Task 微调 - 标准 CoT")
    print("=" * 60)
    print(f"随机种子: {args.seed}")
    print(f"模型: {args.model_path}")
    print(f"训练数据: {args.train_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练模式: {args.train_mode}")
    print(f"Prompt: {COT_PROMPT_PATH}")
    print("=" * 60)

    train_data = load_jsonl(args.train_file)
    print(f"训练集大小: {len(train_data)}")

    model, tokenizer = setup_model(args.model_path, args.output_dir, DEFAULT_CONFIG)

    dataset = prepare_dataset(
        args.train_file, tokenizer, args.max_seq_length,
        use_cot_template=True,
        train_mode=args.train_mode,
    )

    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        seed=args.seed,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=DEFAULT_CONFIG["warmup_steps"],
        logging_steps=DEFAULT_CONFIG["logging_steps"],
        save_steps=DEFAULT_CONFIG["save_steps"],
        save_total_limit=DEFAULT_CONFIG["save_total_limit"],
        max_grad_norm=1.0,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    if not args.skip_eval and args.eval_samples > 0:
        print("\n训练前评估...")
        run_train_eval(model, tokenizer, train_data, args.eval_samples, "train_before",
                       args.output_dir, train_mode=args.train_mode, use_cot_template=True)
    else:
        print("\n跳过训练前评估")

    print("\n开始训练...")
    trainer.train()

    print(f"\n保存模型到: {args.output_dir}/final_model")
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")

    if not args.skip_eval and args.eval_samples > 0:
        print("\n训练后评估...")
        run_train_eval(model, tokenizer, train_data, args.eval_samples, "train_after",
                       args.output_dir, train_mode=args.train_mode, use_cot_template=True)

    print("\n训练完成!")


if __name__ == "__main__":
    main()
