"""
T-CoT风格推理脚本 - Qwen2.5版本
使用T-CoT统一格式 (与Llama版本一致)
"""

import argparse
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm



PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:"
)


def generate(args, model, tokenizer, data):
    """T-CoT统一格式推理"""
    prompts = []
    answers = []
    for item in data:
        question = item.get('query', item.get('instruction', ''))
        answer = item.get('response', item.get('output', ''))
        prompt = PROMPT_TEMPLATE.format(instruction=question)
        prompts.append(prompt)
        answers.append(answer)

    results = []
    max_samples = args.max_samples if args.max_samples > 0 else len(prompts)

    for idx in tqdm(range(max_samples), desc="Generating"):
        inputs = tokenizer(prompts[idx], return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else 0.0,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_length:]
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        results.append({
            'type': 'gsm8k',
            'query': prompts[idx],
            'output': generated_text,
            'answer': answers[idx]
        })

        if idx < 3:
            print(f"\n--- 样本 {idx} ---")
            print(f"生成: {generated_text[:200]}...")

    return results


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 T-CoT推理")
    parser.add_argument('--model', type=str, required=True, help="模型路径")
    parser.add_argument('--data_file', type=str, required=True, help="测试数据文件")
    parser.add_argument('--output_file', type=str, default="model_response.jsonl", help="输出文件")
    parser.add_argument('--max_samples', type=int, default=0, help="最大样本数(0=全部)")
    parser.add_argument('--temperature', type=float, default=0.0, help="温度")
    parser.add_argument('--top_p', type=float, default=1.0, help="top_p")
    parser.add_argument('--max_tokens', type=int, default=512, help="最大生成长度")

    args = parser.parse_args()


    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()


    print(f"Loading data from {args.data_file}...")
    if args.data_file.endswith('.jsonl'):
        with open(args.data_file, 'r') as f:
            data = [json.loads(line) for line in f]
    elif args.data_file.endswith('.json'):
        with open(args.data_file, 'r') as f:
            data = json.load(f)
    print(f"Total samples: {len(data)}")


    results = generate(args, model, tokenizer, data)


    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Done! Generated {len(results)} responses.")


if __name__ == "__main__":
    main()
