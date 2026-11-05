"""
Atom Task 推理脚本 - vLLM 加速版本。

通过 --prompt_file 可切换 CoT / No-CoT prompt 模板：
- 默认使用 No-CoT 模板（直接输出最终答案）
- CoT 实验请传入标准 CoT prompt（含 Example）
"""

import argparse
import json
import os
import re
import shutil
import tempfile
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm



NOCOT_PROMPT_TEMPLATE = """You are given a sequence of operations to perform on a symbol list.
Definitions of atomic operations:
1. Reverse: Reverse the sequence order.
2. Substitute: Apply pairwise swaps {{α↔β, γ↔δ, ε↔ω}}.
3. Duplicate: Copy each symbol consecutively (e.g., αβγ → ααββγγ).
4. Deduplicate: Remove consecutive duplicate elements.
5. Rotate: Move the first element to the end.

Now solve this problem:
Input Sequence: [{input_sequence}]
Task Chain: [{task_chain}]

Provide the final answer directly without showing reasoning steps."""


def format_prompt(input_sequence: str, task_chain: str, prompt_template: str = None) -> str:
    """构建prompt。

    Args:
        prompt_template: 若提供，则使用该模板（支持 --prompt_file）；
                         否则使用默认的 No-CoT 模板。
    """
    prompt = prompt_template if prompt_template else NOCOT_PROMPT_TEMPLATE
    prompt = prompt.replace('{input_sequence}', input_sequence)
    prompt = prompt.replace('{task_chain}', task_chain)
    return prompt


def extract_answer(text: str) -> str:
    """
    从模型输出中提取最终答案，与 test_atom.py 的 extract_answer 保持一致
    处理 </s> 等特殊token
    """

    text = re.sub(r'<BOS>|<EOS>|<\|endoftext\|>|</?s>|<\|eos\|>', '', text)
    text = text.strip()

    if "Final Answer:" in text:
        answer = text.split("Final Answer:")[-1].strip()
        answer = answer.strip('"').strip("'")
        answer = answer.split('\n')[0].strip()
        if answer.startswith('[') and answer.endswith(']'):
            answer = answer[1:-1]
        return answer

    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip()
        answer = answer.strip('"').strip("'")
        answer = answer.split('\n')[0].strip()
        if answer.startswith('[') and answer.endswith(']'):
            answer = answer[1:-1]
        return answer

    if "Output:" in text:
        parts = text.split("Output:")
        if len(parts) > 1:
            answer = parts[-1].strip()
            answer = answer.split('\n')[0].strip()
            if answer.startswith('[') and answer.endswith(']'):
                answer = answer[1:-1]
            return answer

    return text.strip()


def is_lora_model(model_path: str) -> bool:
    """检查是否为LoRA adapter目录"""
    return os.path.isfile(os.path.join(model_path, "adapter_model.safetensors"))


def merge_lora_model(model_path: str) -> str:
    """合并LoRA adapter到基础模型，返回merged模型路径"""
    print(f"检测到LoRA adapter，正在合并...")


    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)

    base_model_path = adapter_config.get("base_model_name_or_path", "")
    if not base_model_path or not os.path.exists(base_model_path):
        raise ValueError(f"无法找到base model: {base_model_path}")

    print(f"  Base model: {base_model_path}")


    print(f"  加载基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )


    print(f"  合并LoRA权重...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()


    merged_dir = tempfile.mkdtemp(prefix="merged_")
    merged_path = os.path.join(merged_dir, "merged_model")
    print(f"  保存merged模型到: {merged_path}")


    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        temperature=1.0,
        do_sample=False,
        max_new_tokens=1024,
    )
    model.generation_config = gen_config

    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)


    del model, base_model
    import torch
    torch.cuda.empty_cache()

    return merged_path, merged_dir, base_model_path


def generate_vllm(args):
    """vLLM批量推理"""


    prompt_template = None
    if getattr(args, 'prompt_file', None):
        print(f"加载 prompt 模板: {args.prompt_file}")
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()


    print(f"Loading data from {args.data_file}...")
    if args.data_file.endswith('.jsonl'):
        with open(args.data_file, 'r') as f:
            data = [json.loads(line) for line in f]
    elif args.data_file.endswith('.json'):
        with open(args.data_file, 'r') as f:
            data = json.load(f)
    print(f"Total samples: {len(data)}")


    prompts = []
    answers = []
    for item in data:

        instruction = item.get('instruction', '')
        output = item.get('output', '')


        if instruction:
            lines = instruction.split('\n')
            input_seq = ""
            task_chain = ""
            for line in lines:
                if 'Input Sequence:' in line:
                    input_seq = line.split('Input Sequence:')[-1].strip()
                    input_seq = input_seq.strip('[]')
                elif 'Task Chain:' in line:
                    task_chain = line.split('Task Chain:')[-1].strip()
        else:
            input_seq = item.get('input', '')
            task_chain = item.get('task_chain_readable', item.get('task_chain', ''))


        answer = extract_answer(output)

        prompts.append(format_prompt(input_seq, task_chain, prompt_template))
        answers.append(answer)

    max_samples = args.max_samples if args.max_samples > 0 else len(prompts)
    prompts = prompts[:max_samples]
    answers = answers[:max_samples]

    print(f"开始vLLM推理，{len(prompts)}个样本批量处理...")


    model_path = args.model
    base_tokenizer_path = model_path
    merged_dir = None


    if is_lora_model(model_path):
        model_path, merged_dir, base_tokenizer_path = merge_lora_model(model_path)
        print(f"使用merged模型: {model_path}")

    try:

        print(f"加载tokenizer并应用chat template...")
        tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            formatted_prompts.append(formatted_text)


        print(f"Loading vLLM model from {model_path}...")
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="half",
            max_model_len=4096,
            max_num_seqs=256,
            max_num_batched_tokens=4096,
        )



        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_tokens,
            stop=None,
        )


        outputs = llm.generate(formatted_prompts, sampling_params)

        results = []
        for idx, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            results.append({
                'predict': generated_text,
                'label': answers[idx]
            })

            if idx < 3:
                print(f"\n--- 样本 {idx} ---")
                print(f"问题: {prompts[idx][:100]}...")
                print(f"生成: {generated_text[:200]}...")
                print(f"答案: {answers[idx]}")


        print(f"\nSaving results to {args.output_file}...")
        os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else '.', exist_ok=True)
        with open(args.output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"Done! Generated {len(results)} responses.")
    finally:
        if merged_dir and os.path.exists(merged_dir):
            print(f"清理临时merged模型目录: {merged_dir}")
            shutil.rmtree(merged_dir)


def main():
    parser = argparse.ArgumentParser(description="Atom Task No-CoT 推理 (vLLM加速)")
    parser.add_argument('--model', type=str, required=True, help="模型路径 (LoRA adapter或merged模型)")
    parser.add_argument('--data_file', type=str, required=True, help="测试数据文件 (JSONL格式)")
    parser.add_argument('--output_file', type=str, default="model_response.jsonl", help="输出文件")
    parser.add_argument('--max_samples', type=int, default=0, help="最大样本数(0=全部)")
    parser.add_argument('--temperature', type=float, default=0.0, help="温度")
    parser.add_argument('--top_p', type=float, default=1.0, help="top_p")
    parser.add_argument('--max_tokens', type=int, default=1024, help="最大生成长度")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8,
                       help="GPU内存利用率 (默认0.8)")
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help="张量并行大小 (多GPU时使用)")
    parser.add_argument('--prompt_file', type=str, default=None,
                       help="可选：prompt 模板路径（CoT 实验请传入标准 CoT prompt）")

    args = parser.parse_args()
    generate_vllm(args)


if __name__ == "__main__":
    main()