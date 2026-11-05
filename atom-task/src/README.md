# Shared Code (src/)

Code and prompt templates shared across experiments.

## Files

| File | Purpose |
|------|---------|
| `data_generator.py` | Atom-Task data-generation core (5 atomic operations and composition) |
| `finetune_qwen_atom.py` | Standard CoT fine-tuning (LoRA + Qwen3-4B) |
| `finetune_qwen_atom_nocot.py` | No-CoT fine-tuning (direct `Final Answer`) |
| `finetune_qwen_atom_ablation.py` | State-tracking ablation fine-tuning (`--ablation_mode`) |
| `gen_response_vllm.py` | vLLM batch inference (`--prompt_file` switches CoT/No-CoT prompt) |
| `eval_accuracy.py` | Accuracy evaluation |
| `export_samples.py` | Prediction-sample export |
| `prompts/` | Prompt templates |
| `prompts/prompt_cot_atom.txt` | Standard CoT prompt (with example) |
| `prompts/prompt_nocot_atom.txt` | No-CoT prompt |
| `prompts/prompt_cot_atom_ablation.txt` | State-tracking ablation prompt |

## Notes

- Each fine-tuning script accepts `--prompt_file` to specify a prompt template (defaults to the matching template in `src/prompts/`).
- Run scripts reference this directory via `${ROOT_DIR}/src/...`.
- `data_generator.py` is the original module. `finetune_qwen_atom.py` is a release reconstruction; replace with the original if available (interface-compatible).
