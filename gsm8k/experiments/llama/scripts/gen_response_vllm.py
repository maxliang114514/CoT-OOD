"""
T-CotMechanism风格推理脚本 - vLLM加速版本
使用vLLM的PagedAttention + Continuous Batching进行高效推理
自动处理LoRA adapter的合并
"""

import argparse
import json
import os
import shutil
import tempfile
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm



PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response: Let's think step by step."
)


def format_prompt(question: str) -> str:
    """构建T-CotMechanism格式的prompt"""
    return PROMPT_TEMPLATE.format(instruction=question)


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
        trust_remote_code=True
    )


    print(f"  合并LoRA权重...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()


    merged_dir = tempfile.mkdtemp(prefix="merged_")
    merged_path = os.path.join(merged_dir, "merged_model")
    print(f"  保存merged模型到: {merged_path}")


    if model.generation_config is not None:
        gen_cfg = model.generation_config
        gen_cfg.temperature = None
        gen_cfg.top_p = None
        gen_cfg.do_sample = False

    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)


    del model, base_model
    import torch
    torch.cuda.empty_cache()

    return merged_path, merged_dir


def generate_vllm(args):
    """vLLM批量推理"""


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
        question = item.get('query', item.get('instruction', ''))
        answer = item.get('response', item.get('output', ''))
        prompts.append(format_prompt(question))
        answers.append(answer)

    max_samples = args.max_samples if args.max_samples > 0 else len(prompts)
    prompts = prompts[:max_samples]
    answers = answers[:max_samples]

    print(f"开始vLLM推理，{len(prompts)}个样本批量处理...")


    model_path = args.model
    merged_dir = None


    if is_lora_model(model_path):
        model_path, merged_dir = merge_lora_model(model_path)
        print(f"使用merged模型: {model_path}")

    try:

        print(f"Loading vLLM model from {model_path}...")
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="half",
        )


        sampling_params = SamplingParams(
            temperature=args.temperature if args.temperature > 0 else 0.0,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop=None,
        )


        outputs = llm.generate(prompts, sampling_params)

        results = []
        for idx, output in enumerate(outputs):
            generated_text = output.outputs[0].text

            results.append({
                'type': 'gsm8k',
                'query': prompts[idx],
                'output': generated_text,
                'answer': answers[idx]
            })

            if idx < 3:
                print(f"\n--- 样本 {idx} ---")
                print(f"问题: {prompts[idx][:100]}...")
                print(f"生成: {generated_text[:200]}...")


        print(f"\nSaving results to {args.output_file}...")
        with open(args.output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"Done! Generated {len(results)} responses.")
    finally:
        if merged_dir and os.path.exists(merged_dir):
            print(f"清理临时merged模型目录: {merged_dir}")
            shutil.rmtree(merged_dir)


def main():
    parser = argparse.ArgumentParser(description="T-CotMechanism风格推理 (vLLM加速)")
    parser.add_argument('--model', type=str, required=True, help="模型路径")
    parser.add_argument('--data_file', type=str, required=True, help="测试数据文件")
    parser.add_argument('--output_file', type=str, default="model_response.jsonl", help="输出文件")
    parser.add_argument('--max_samples', type=int, default=0, help="最大样本数(0=全部)")
    parser.add_argument('--temperature', type=float, default=0.0, help="温度")
    parser.add_argument('--top_p', type=float, default=1.0, help="top_p")
    parser.add_argument('--max_tokens', type=int, default=512, help="最大生成长度")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                       help="GPU内存利用率 (默认0.9)")
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help="张量并行大小 (多GPU时使用)")

    args = parser.parse_args()
    generate_vllm(args)


if __name__ == "__main__":
    main()