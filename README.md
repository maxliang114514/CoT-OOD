# Understanding the Out-of-Distribution Generalization of Chain-of-Thought Reasoning in LLMs(COT-OOD)

Companion code for *"Understanding the Out-of-Distribution Generalization of Chain-of-Thought Reasoning in LLMs"* (NeurIPS 2026). Paper: http://arxiv.org/abs/2611.06438

This repository contains the full experimental implementation of the paper across two benchmark suites:

| Directory | Description | Paper reference |
|-----------|-------------|-----------------|
| [`atom-task/`](./atom-task/) | Symbolic compositional reasoning benchmark (Qwen3-4B) | Section 5, Appendix C.1 |
| [`gsm8k/`](./gsm8k/) | Real-world mathematical reasoning (Llama2-7B / Qwen2.5-3B) | Appendix C.2 |

## Overview

The paper studies Chain-of-Thought (CoT) reasoning as **compositional out-of-distribution (OOD) generalization**: the model learns only in-distribution atomic step transitions and composes a shared single-step predictor to solve unseen multi-step tasks. The theory gives a generalization gap of O(sqrt(T/n)), sublinear in reasoning depth T.

Experiments verify:

- **Atom-Task**: short-to-long generalization, operation ablation, state-tracking ablation, scaling laws (depth/sample/context), and noise robustness.
- **GSM8K**: CoT vs. No-CoT, reasoning-noise robustness, and topic / complexity / answer-scale distribution shifts.

## Layout

```
.
├── README.md
├── atom-task/          # Atom-Task experiments (see atom-task/README.md)
└── gsm8k/              # GSM8K experiments (see gsm8k/README.md)
```

Each sub-project is self-contained, with its own datasets, models, shared code (`src/`), and experiment directories. See the README inside each for setup and usage.

## Data

- Base datasets are included in `atom-task/datasets/` and `gsm8k/datasets/` where small enough.
- `gsm8k/datasets/metamath/MetaMathQA-395K.json` (~395 MB) exceeds GitHub's file-size limit and is **excluded**; download it from [MetaMath](https://meta-math.github.io/) or generate the OOD data with `gsm8k/experiments/*/ood/run_ood_pipeline.sh data_only`.

## Models

- Place base models in `models/` under each sub-project (`atom-task/models/`, `gsm8k/models/`), or set `MODEL_PATH`.
- Required base models: Qwen3-4B (Atom-Task), Llama2-7B and Qwen2.5-3B (GSM8K).

## Citation

Please cite the paper if you use this code.
```
@article{liang2026understanding,
  title={Understanding the Out-of-Distribution Generalization of Chain-of-Thought Reasoning in LLMs},
  author={Liang, Nuo-Jing and Yuan, Xiao-Tong and Zhou, Pan},
  journal={arXiv preprint arXiv:2611.06438},
  year={2026}
}
```
