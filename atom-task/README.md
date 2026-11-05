# Understanding the Out-of-Distribution Generalization of Chain-of-Thought Reasoning in LLMs

Companion code for the Atom-Task experiments in *"Understanding the Out-of-Distribution Generalization of Chain-of-Thought Reasoning in LLMs"* (NeurIPS 2026).

The paper studies Chain-of-Thought (CoT) reasoning through the lens of **compositional out-of-distribution (OOD) generalization**: during training the model only sees atomic single-step data, yet at test time it must repeatedly compose a shared single-step predictor to solve unseen multi-step compositional tasks. The theoretical generalization gap is O(√(T/n)) — sublinear in reasoning depth T due to predictor sharing — and in the next-token-prediction setting becomes O(k√(T/mn) + k√(T/n) + kT/m).

This repository implements all Atom-Task experiments from the paper:

1. **Short-to-Long Generalization** and the **CoT vs. No-CoT** comparison (Table 1)
2. **Generalization to Unseen Combinations** (operation-wise ablation, Table 2)
3. **Effect of State Tracking in CoT** (Table 3, Figure 2)
4. **Scaling laws** for reasoning depth T, sample size n, and context length m (Figure 3, Tables 6/7)
5. **Robustness to training noise** (Table 8)

## Paper tables

| Paper table | Directory | Content |
|---|---|---|
| Table 1(a) | experiments/table1_main_cot/ | CoT main: 1-step / 3-step / Mix3+5 training, tested at 3/5/7/9 steps |
| Table 1(b) | experiments/table1_main_nocot/ | No-CoT counterpart |
| Table 2(a) | experiments/table1_main_cot/ (
un_ablation.sh) | CoT, one operation excluded |
| Table 2(b) | experiments/table2_ablation_nocot/ | No-CoT, one operation excluded |
| Table 3 | experiments/table3_state_tracking/ | Format ablation: explicit state tracking vs. masked state |
| Table 6 | experiments/table6_7_scaling/ | Training-sample scaling (n) |
| Table 7 | experiments/table6_7_scaling/ | Inference-time examples |
| Table 8 | experiments/table8_noise/ | Robustness to training noise |

## Repository Layout

```
.
├── README.md
├── .gitignore
├── datasets/                  # Shared base datasets
├── models/                    # Base model location (see models/README.md)
├── src/                       # Shared code (data generation / fine-tuning / inference / evaluation)
└── experiments/
    ├── t_scaling/             # T-scaling (main) + operation ablation (CoT)
    ├── t_scaling_nocot/       # Main experiment, No-CoT control
    ├── nm_scaling/            # n / m scaling
    ├── noise_robust/          # Noise robustness
    ├── state_tracking/        # State-tracking ablation
    └── ablation_nocot/        # Operation ablation, No-CoT control
```

## Experiment to Paper Mapping

| Directory | Paper Reference | Description |
|-----------|----------------|-------------|
| `experiments/table1_main_cot/` | Figure 3a, Table 1a, Table 2a | Main experiment (T-scaling / reasoning depth) and operation ablation (CoT) |
| `experiments/table1_main_nocot/` | Table 1b | Main experiment, No-CoT control (3 seeds) |
| `experiments/table6_7_scaling/` | Figure 3b/3c, Tables 6/7 | Sample-size (n) and context-length (m) scaling |
| `experiments/table8_noise/` | Table 8 | Robustness to reasoning-step noise |
| `experiments/table3_state_tracking/` | Table 3, Figure 2 | CoT format ablation (mask Task_Chain / Input_Sequence) |
| `experiments/table2_ablation_nocot/` | Table 2b | Operation ablation, No-CoT control |

## Datasets

All experiments share the base datasets in `datasets/`:

| File | Description | Size |
|------|-------------|------|
| `train_1step.jsonl` | Single-step training | 5,000 |
| `train_3step.jsonl` | Three-step training | 1,667 |
| `train_mixed_3_5step.jsonl` | Mixed 3+5 step training | 1,333 |
| `ablation_no_{reverse,substitute,duplicate,deduplicate,rotate}.jsonl` | Operation-excluded ablation sets | 5,000 each |
| `test_{3,5,7,9}step.jsonl` | Held-out test sets | 500 each |

See `datasets/README.md` for the record format.

## Models

Base model: **Qwen3-4B**. Place the weights in `models/qwen3-4b/`, or override with `MODEL_PATH`:

```bash
export MODEL_PATH=/path/to/Qwen3-4B
```

Fine-tuning: LoRA (rank=16, alpha=32), 3 epochs, lr=5e-5, max_new_tokens=2048. Inference uses vLLM (temperature=0.0).

## Usage

All scripts resolve project paths relative to the repository root.

```bash
# Main experiment (T-scaling, CoT) — train_3step + train_mixed_3_5step, 3 seeds
cd experiments/table1_main_cot/scripts
CUDA_VISIBLE_DEVICES=0 bash run_scaling.sh all_3step full

# Main experiment, No-CoT control
cd experiments/table1_main_nocot/scripts
CUDA_VISIBLE_DEVICES=0 bash run_scaling.sh all_3step full

# Operation ablation (CoT / No-CoT)
cd experiments/table1_main_cot/scripts && bash run_ablation.sh full        # CoT
cd experiments/table2_ablation_nocot/scripts && bash run_pipeline.sh full   # No-CoT

# n / m scaling
cd experiments/table6_7_scaling/scripts && bash run_nm_scaling.sh all

# Noise robustness
cd experiments/table8_noise/scripts && bash run_pipeline.sh all

# State-tracking ablation
cd experiments/table3_state_tracking/scripts && bash run_pipeline.sh full
```

Results are written to each experiment's `results/`; trained LoRA adapters to `experiments/<name>/output/` (git-ignored).

## Requirements

```bash
pip install torch transformers peft datasets vllm
```

## Notes

- `src/data_generator.py` is the original module from the author's data package.
- `src/finetune_qwen_atom.py` is a release reconstruction based on the paper's CoT format; replace with the original if you have it (interface-compatible).
