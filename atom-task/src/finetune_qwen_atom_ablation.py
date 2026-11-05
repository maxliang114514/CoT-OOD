#!/usr/bin/env python3
"""
Atom Task (符号替换任务) 微调脚本 - 消融实验版本
训练时生成的 output 思维链去掉当前 task_chain 的记录，
但 Input_Sequence 正常显示（消融实验中保留input信息）

消融格式（去掉 Task_Chain 记录 + 保留 Input_Sequence）：
Step 1: [Duplicate] Task_Chain: [......] | Input_Sequence: [xxx] | → Output_Sequence: [xxx] | Task_Chain: [......]
Step 2: [Reverse] Task_Chain: [......] | Input_Sequence: [xxx] | → Output_Sequence: [xxx] | Task_Chain: [......]
Final Answer: [xxx]
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

EVAL_SAMPLES = 100  # 训练前后各测试的样本数

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


# CoT Prompt 模板路径（默认标准格式；消融实验请通过 --prompt_file 指定）
COT_PROMPT_PATH = os.path.join(_SRC_DIR, 'prompts', 'prompt_cot_atom.txt')


def load_cot_template(template_path: str = None) -> str:
    """加载CoT提示模板"""
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


def convert_to_cot_format_ablation(sample: dict) -> str:
    """将sample的output转换为消融格式 - 保持消融信息量，对齐原始格式长度

    消融格式（带padding，Task_Chain用......消融，但Input_Sequence正常显示）：
    Step 1: [Duplicate] Task_Chain: [......, ......] | Input_Sequence: [xxx] | → Output_Sequence: [xxx] | Task_Chain: [......, ......]
    Step 2: [Reverse] Task_Chain: [......] | Input_Sequence: [xxx] | → Output_Sequence: [xxx] | Task_Chain: [......]
    Final Answer: [xxx]
    """
    input_text = sample.get('input', '')
    output = sample.get('output', '')
    task_chain_str = sample.get('task_chain_readable', sample.get('task_chain', ''))
    reasoning_steps = sample.get('reasoning_steps', [])

    # 解析task_chain为列表
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
        current_task = step_info.get('task', task_chain[i] if i < len(task_chain) else '')
        remaining_tasks = task_chain[i + 1:]

        # 生成...... placeholder，保留与原始格式相同的长度
        # 每个......代表一个被消融的任务
        remaining_placeholders = '[' + ', '.join(['......'] * len(remaining_tasks)) + ']' if remaining_tasks else '[]'

        # 消融格式：当前任务 + padding后的完整格式
        # Input_Sequence 正常显示（消融实验只去掉Task_Chain记录，保留input信息）
        step_text = f"Step {i+1}: [{current_task}] Task_Chain: {remaining_placeholders} | Input_Sequence: [{step_input}] | → Output_Sequence: [{step_output}] | Task_Chain: {remaining_placeholders}"
        steps.append(step_text)

    # 最后添加 Final Answer 行
    cot_output = '\n'.join(steps) + f"\nFinal Answer: [{output}]"
    return cot_output


def format_instruction_qwen(sample: dict, tokenizer, use_cot_template: bool = True,
                           train_mode: str = "normal",
                           partial_loss: bool = False, ablation_mode: bool = False) -> str:
    """将样本格式化为Qwen模型的指令格式

    Args:
        ablation_mode: 是否使用消融模式（去掉 Task_Chain 记录 + 保留 Input_Sequence）
    """
    # 如果数据已有完整的instruction字段，直接使用（LlamaFactory格式）
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
        # 如果output已经是CoT格式，直接使用
        if 'Step' not in output:
            # 根据是否是消融模式选择不同格式
            if ablation_mode:
                output = convert_to_cot_format_ablation(sample)
            else:
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


# 全局变量存储训练模式
_TRAIN_MODE = "normal"
_PARTIAL_LOSS = False
_ABLATION_MODE = False

def set_train_mode(mode: str, partial_loss: bool = False, ablation_mode: bool = False):
    """设置训练模式"""
    global _TRAIN_MODE, _PARTIAL_LOSS, _ABLATION_MODE
    _TRAIN_MODE = mode
    _PARTIAL_LOSS = partial_loss
    _ABLATION_MODE = ablation_mode


def tokenize_function(examples):
    """使用全局变量进行tokenize"""
    global _TOKENIZER_PATH, _MAX_LENGTH
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize input
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
                   train_mode: str = "normal", partial_loss: bool = False,
                   ablation_mode: bool = False) -> Dataset:
    print(f"加载训练数据: {train_file}")
    data = load_jsonl(train_file)
    print(f"  总样本数: {len(data)}")
    print(f"  训练模式: {train_mode}")
    print(f"  部分Loss: {partial_loss}")
    print(f"  使用CoT模板: {use_cot_template}")
    print(f"  消融模式 (去掉Task_Chain记录 + 保留Input_Sequence): {ablation_mode}")

    # 设置全局训练模式
    set_train_mode(train_mode, partial_loss, ablation_mode)

    texts = []
    for i, sample in enumerate(data):
        try:
            text = format_instruction_qwen(sample, tokenizer, use_cot_template, train_mode, partial_loss, ablation_mode)
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
    # 直接在主进程中处理所有数据，避免 tokenizer 无法序列化的问题
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for text in texts:
        # Tokenize
        result = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=True,
            return_tensors=None
        )

        tokens = result['input_ids']

        if partial_loss and "Final Answer:" in text:
            # 查找Final Answer的位置
            final_answer_tokens = tokenizer("Final Answer:", add_special_tokens=False)['input_ids']
            content_tokens = tokens[1:-1] if len(tokens) > 2 else tokens

            answer_pos = -1
            for j in range(len(content_tokens) - len(final_answer_tokens) + 1):
                if content_tokens[j:j+len(final_answer_tokens)] == final_answer_tokens:
                    answer_pos = j + 1
                    break

            if answer_pos > 0:
                answer_start_pos = answer_pos + len(final_answer_tokens)
            else:
                # 使用attention_mask找出实际内容的长度
                attn_mask = result['attention_mask']
                actual_len = sum(attn_mask)  # 实际内容长度
                answer_start_pos = int(actual_len * 0.7)  # 假设答案占30%

            # 构建labels - 只对实际内容部分设置
            attn_mask = result['attention_mask']
            actual_len = sum(attn_mask)

            # 前缀部分设为-100
            labels = [-100] * answer_start_pos
            # 答案部分设为token id
            labels.extend(tokens[answer_start_pos:actual_len])
            # padding部分设为-100
            labels.extend([-100] * (len(tokens) - actual_len))
        else:
            labels = list(tokens)

        all_input_ids.append(tokens)
        all_attention_mask.append(result['attention_mask'])
        all_labels.append(labels)

    dataset = Dataset.from_dict({
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'labels': all_labels
    })

    if partial_loss:
        print(f"  Input_ids长度: {len(dataset[0]['input_ids'])}")
        print(f"  Labels长度: {len(dataset[0]['labels'])}")

    return dataset


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
    """
    从模型输出中提取最终答案
    - normal模式: 直接返回完整输出
    - cot_format/partial_loss模式: 提取Final Answer后的内容（去除首尾方括号）
    """
    text = text.strip()
    text = re.sub(r'<BOS>|<EOS>|<|endoftext|>', '', text)

    if train_mode == "cot_format":
        # 提取 Final Answer 后面的内容
        if "Final Answer:" in text:
            answer = text.split("Final Answer:")[-1].strip()
            answer = answer.strip('"').strip("'")
            answer = answer.split('\n')[0].strip()
            # 去除首尾方括号 [xxx] -> xxx
            if answer.startswith('[') and answer.endswith(']'):
                answer = answer[1:-1]
            return answer

    # normal模式: 直接返回
    return text.strip()


def evaluate_on_train(model, tokenizer, train_data: List[Dict], train_mode: str = "normal",
                      use_cot_template: bool = True,
                      max_new_tokens: int = 1024) -> Tuple[List[str], float]:
    """
    在训练集上评估模型
    比较的是最终的输出序列（不是完整格式）
    """
    predictions = []
    correct = 0

    for sample in tqdm(train_data, desc="评估中"):
        # 构建prompt (使用CoT模板)
        # 如果数据已有完整的instruction字段，直接使用（LlamaFactory格式）
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
            if prompt.endswith('\n'):
                prompt = prompt.rstrip()

        # 原始输出（ground truth）
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

            try:
                content = tokenizer.decode(output_ids, skip_special_tokens=True)
            except:
                content = tokenizer.decode(output_ids, skip_special_tokens=True)

        predictions.append(content)

        # 提取最终答案并比较
        predicted_answer = extract_final_answer(content, train_mode)
        # ground truth 也需要提取 Final Answer 后面的内容
        target_answer = extract_final_answer(target_output, train_mode)

        if predicted_answer.strip() == target_answer.strip():
            correct += 1

    accuracy = correct / len(train_data) if train_data else 0
    return predictions, accuracy


def run_train_eval(model, tokenizer, train_data: List[Dict],
                  eval_samples: int, task_desc: str, output_dir: str,
                  train_mode: str = "normal",
                  use_cot_template: bool = True) -> Dict:
    """在训练集上运行评估"""
    # 随机抽样
    if len(train_data) > eval_samples:
        eval_data = random.sample(train_data, eval_samples)
    else:
        eval_data = train_data

    print(f"\n{task_desc}: 测试 {len(eval_data)} 个样本")

    # 保存评估样本到output_dir下
    eval_file = os.path.join(output_dir, f"eval_samples_{task_desc.replace(' ', '_')}.jsonl")
    save_jsonl(eval_data, eval_file)
    print(f"  评估样本已保存: {eval_file}")

    predictions, accuracy = evaluate_on_train(
        model, tokenizer, eval_data, train_mode,
        use_cot_template
    )

    print(f"\n{task_desc} 结果:")
    print(f"  准确率: {accuracy*100:.2f}% ({int(accuracy*len(eval_data))}/{len(eval_data)})")

    print(f"\n  示例 (前3个):")
    for i in range(min(3, len(eval_data))):
        print(f"    样本{i+1}:")

        # 生成模型看到的完整输入（只包含prompt，不包含目标输出）
        sample = eval_data[i]

        # 如果数据已有完整的instruction字段，直接使用（LlamaFactory格式）
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

        # 使用 Qwen 的格式
        bos_token = tokenizer.bos_token if tokenizer.bos_token else ""
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        full_input = f"{bos_token}{im_start}user\n{prompt}{im_end}\n{im_start}assistant\n"

        # 完整显示
        print(f"      完整输入:")
        for line in full_input.split('\n'):
            print(f"        {line}")

        print(f"      原始输入: {eval_data[i]['input'][:50]}...")
        print(f"      目标: {eval_data[i]['output']}")
        pred = predictions[i][:100] if predictions[i] else '(空)'
        print(f"      预测: {pred}...")
        pred_answer = extract_final_answer(predictions[i], train_mode)
        print(f"      预测答案: {pred_answer}")
        print(f"      {'✓' if pred_answer.strip() == eval_data[i]['output'].strip() else '✗'}")

    return {
        'task': task_desc,
        'num_samples': len(eval_data),
        'accuracy': accuracy,
        'predictions': predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen Atom Task微调 - 消融实验版本 (去掉Task_Chain记录 + 保留Input_Sequence)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--train_file", type=str, default=f"{BASE_DIR}/train.jsonl", help="训练数据")
    parser.add_argument("--use_composition_train", action="store_true",
                       help="使用3步组合任务训练集 (train_composition_3step.jsonl)")
    parser.add_argument("--output_dir", type=str, default="./output_qwen_atom", help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="batch大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--train_mode", type=str, default="normal",
                       choices=["normal", "cot_format"],
                       help="训练模式: normal=正常训练, cot_format=Step+Final Answer格式")
    parser.add_argument("--partial_loss", action="store_true",
                       help="在cot_format基础上，只对Final Answer计算Loss（需配合--train_mode cot_format使用）")
    parser.add_argument("--eval_samples", type=int, default=EVAL_SAMPLES,
                        help="训练前后评估的样本数")
    parser.add_argument("--skip_eval", action="store_true",
                       help="跳过训练前后评估")
    parser.add_argument("--ablation_mode", action="store_true",
                       help="消融模式：训练时生成的output思维链去掉Task_Chain记录，但Input_Sequence正常显示")
    parser.add_argument("--prompt_file", type=str, default=COT_PROMPT_PATH,
                       help="CoT prompt 模板路径（消融实验请指向 prompt_cot_atom_ablation.txt）")

    args = parser.parse_args()
    set_seed(args.seed)

    global COT_PROMPT_PATH
    COT_PROMPT_PATH = args.prompt_file

    print("=" * 60)
    print("Qwen Atom Task 微调 - 消融实验版本")
    print("=" * 60)
    print(f"随机种子: {args.seed}")
    print(f"模型: {args.model_path}")
    print(f"Prompt: {COT_PROMPT_PATH}")

    # 处理训练集选择
    if args.use_composition_train:
        train_file = f"{BASE_DIR}/train_composition_3step.jsonl"
        print(f"训练数据: {train_file} (3步组合任务)")
    else:
        train_file = args.train_file
        print(f"训练数据: {train_file}")

    print(f"输出目录: {args.output_dir}")
    print(f"训练模式: {args.train_mode}")
    print(f"评估样本数: {args.eval_samples}")
    print(f"消融模式 (去掉Task_Chain记录 + 保留Input_Sequence): {args.ablation_mode}")

    # 加载训练数据
    train_data = load_jsonl(train_file)
    print(f"训练集大小: {len(train_data)}")

    model, tokenizer = setup_model(args.model_path, args.output_dir, DEFAULT_CONFIG)

    dataset = prepare_dataset(
        train_file, tokenizer, args.max_seq_length,
        use_cot_template=True,
        train_mode=args.train_mode,
        partial_loss=args.partial_loss,
        ablation_mode=args.ablation_mode
    )

    # 启用 gradient checkpointing 以减少内存使用
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 启用 gradient checkpointing
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=DEFAULT_CONFIG["warmup_steps"],
        logging_steps=DEFAULT_CONFIG["logging_steps"],
        save_steps=DEFAULT_CONFIG["save_steps"],
        save_total_limit=DEFAULT_CONFIG["save_total_limit"],
        max_grad_norm=1.0,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",  # 禁用wandb
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 训练前评估（使用训练集）
    if not args.skip_eval and args.eval_samples > 0:
        print("\n" + "=" * 60)
        print("训练前评估 (使用训练集)")
        print("=" * 60)

        eval_before = run_train_eval(
            model, tokenizer, train_data,
            args.eval_samples, "train_before",
            args.output_dir,
            train_mode=args.train_mode,
            use_cot_template=True
        )

        print(f"\n训练前准确率: {eval_before['accuracy']*100:.2f}%")
    else:
        print("\n跳过训练前评估")
        eval_before = None

    # 释放训练前评估的内存
    print("\n释放评估用内存...")
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("内存已释放")

    # 重新加载模型用于训练
    print("重新加载模型...")
    model, tokenizer = setup_model(args.model_path, args.output_dir, DEFAULT_CONFIG)

    # 启用 gradient checkpointing 以减少内存使用
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    model.gradient_checkpointing_enable()

    # 重新创建 TrainingArguments 和 Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    trainer.train()

    print(f"\n保存模型到: {args.output_dir}/final_model")
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")

    # 训练后评估（使用训练集）
    if not args.skip_eval and args.eval_samples > 0:
        print("\n" + "=" * 60)
        print("训练后评估 (使用训练集)")
        print("=" * 60)

        eval_after = run_train_eval(
            model, tokenizer, train_data,
            args.eval_samples, "train_after",
            args.output_dir,
            train_mode=args.train_mode,
            use_cot_template=True
        )

        print(f"\n训练后准确率: {eval_after['accuracy']*100:.2f}%")

        if eval_before:
            print(f"\n准确率提升: {(eval_after['accuracy'] - eval_before['accuracy'])*100:.2f}%")
    else:
        print("\n跳过训练后评估")

    print("\n训练完成!")


if __name__ == "__main__":
    main()