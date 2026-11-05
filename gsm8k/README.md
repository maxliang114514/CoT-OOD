# GSM8K Experiments (CoT Training and OOD Generalization)

Companion code for the GSM8K experiments (Appendix C.2) of *"Understanding the Out-of-Distribution Generalization of Chain-of-Thought Reasoning in LLMs"* (NeurIPS 2026).

The experiments study how CoT training affects mathematical reasoning and three OOD generalization capabilities:

1. **CoT vs. No-CoT**: benefit of intermediate step supervision (Table 9);
2. **Noise robustness**: reasoning-step noise (delete step / wrong number), p∈{0.05, 0.10, 0.20, 0.50} (Table 11);
3. **Distribution shift (OOD)**: topic / complexity / answer-scale shifts (Table 10);
4. **Stepwise reasoning**: decomposing multi-step problems into atomic samples.

Two base models are used — **Llama2-7B** and **Qwen2.5-3B** — both LoRA fine-tuned on MetaMathQA.

## Paper tables

| Paper table | Where |
|---|---|
| Table 9 (baseline, CoT vs. No-CoT) | experiments/{llama,qwen}/baseline/, experiments/{llama,qwen}/cot_no_cot/ |
| Table 10 (distribution shift) | experiments/{llama,qwen}/ood/ (ilter_distribution_shift.py --shift_mode {domain,complexity,magnitude}) |
| Table 11 (training-time reasoning noise) | experiments/{llama,qwen}/ood/ (dd_noise.py) |

## Repository Layout

```
.
├── README.md                       # This document (single experiment guide)
├── .gitignore
├── datasets/                       # Shared datasets
│   ├── metamath/
│   │   ├── MetaMathQA-395K.json    # Training data (395K)
│   │   └── test.json               # MetaMathQA test (reference)
│   └── gsm8k/
│       ├── test.jsonl              # GSM8K test set (1,319 samples, used by scripts)
│       ├── train.parquet           # Raw GSM8K train (reference)
│       └── test.parquet            # Raw GSM8K test (reference)
├── models/                         # Base model location (see models/README.md)
└── experiments/
    ├── llama/                      # Llama2-7B experiments
    │   ├── baseline/               #   Baseline evaluation
    │   ├── cot_no_cot/             #   CoT / No-CoT training
    │   ├── ood/                    #   OOD experiments (data/results/scripts)
    │   └── scripts/                #   Inference scripts
    └── qwen/                       # Qwen2.5-3B experiments
        ├── baseline/
        ├── ood/
        └── (training/inference scripts)
```

## Datasets

| Purpose | Path |
|---------|------|
| Training | `datasets/metamath/MetaMathQA-395K.json` |
| Test | `datasets/gsm8k/test.jsonl` (GSM8K test, 1,319 samples) |
| Raw GSM8K | `datasets/gsm8k/train.parquet`, `test.parquet` (reference) |
| MetaMathQA test | `datasets/metamath/test.json` (reference) |

## Models

| Model | Base path | Fine-tuned output |
|-------|-----------|-------------------|
| Llama2-7B | `models/llama2-7b/` | `experiments/llama/output/cot_llama2_7b_metamath/` |
| Qwen2.5-3B | `models/qwen2.5-3b/` | `experiments/qwen/output/cot_qwen2.5_3b_metamath/` |

Place base weights in `models/`, or override with `MODEL_PATH`.

## Experiment Scale (matches the paper)

> The authoritative pipeline is `run_ood_pipeline.sh`; all results in Tables 9/10/11 come from it.
> Each experiment trains on **30,000** samples (consistent with the paper).

| Experiment | Data file | Data size | Training samples |
|------------|-----------|-----------|------------------|
| Baseline (untrained) | GSM8K test | 1,319 | — |
| CoT training (noise0) | `metamath_noise0.json` | 395,000 | **30,000** |
| No-CoT control | `metamath_nocot.json` | 30,000 | **30,000** |
| Noise 5/10/20/50% | `metamath_noise{5,10,20,50}.json` | 395,000 (full, noised) | **30,000** |
| Stepwise | `metamath_stepwise.jsonl` | 30,000 originals → 69,937 atomic | all (cap 210,000) |
| Distribution shift (domain/complexity/magnitude/all) | `metamath_ood_*.json` | 100,000 (top 30% shifted) | **30,000** |

> Noise/OOD data files are generated from the full MetaMathQA (395K noised; 100K shifted), and training takes the first 30,000 samples.
> A separate `train_llama2_7b_metamath.sh` / `train_qwen2.5_3b_metamath.sh` (100K "original T-CoT" reproduction) is **not** reported in the paper; the Table 9 CoT-training numbers come from the 30K standard CoT run.

## Usage

All scripts resolve project paths relative to the repository root.

```bash
# 1. Baseline (untrained)
cd experiments/llama/baseline
CUDA_VISIBLE_DEVICES=0 bash run_baseline.sh

cd experiments/qwen/baseline
CUDA_VISIBLE_DEVICES=0 bash run_baseline_qwen.sh

# 2. OOD experiments (authoritative pipeline, 30K)
cd experiments/llama/ood
CUDA_VISIBLE_DEVICES=0 bash run_ood_pipeline.sh all full        # baseline → nocot → noise → stepwise → ood

# Individual experiments
CUDA_VISIBLE_DEVICES=0 bash run_ood_pipeline.sh nocot full
CUDA_VISIBLE_DEVICES=0 bash run_ood_pipeline.sh noise full
CUDA_VISIBLE_DEVICES=0 bash run_ood_pipeline.sh stepwise full
CUDA_VISIBLE_DEVICES=0 bash run_ood_pipeline.sh ood full

# 3. Test fine-tuned models
cd experiments/llama/cot_no_cot && CUDA_VISIBLE_DEVICES=0 bash run_test_trained.sh
cd experiments/qwen && CUDA_VISIBLE_DEVICES=0 bash run_test_trained.sh
```

> The standalone scripts (`run_noise_exp.sh`, `run_stepwise_exp.sh`, `run_ood_exp.sh`, `run_ood_experiments.py`) use smaller sample sizes and are for quick debugging only — they do **not** reproduce the paper numbers.

## OOD Details

**Noise injection** (Table 11): each sample is selected with probability p; for selected samples, one random reasoning step is corrupted by `delete` or `wrong_number` (50% each).

**Distribution shift** (Table 10): the most shifted 30% of MetaMathQA is selected (`shift_ratio=0.3`, 100K out, train on 30K):

| Shift | Method | Paper name |
|-------|--------|-----------|
| `domain` | Jaccard distance of sample keywords vs. GSM8K high-frequency keywords | Topic Shift |
| `complexity` | Z-score of step count vs. GSM8K mean/std | Complexity Shift |
| `magnitude` | Answer-magnitude distribution mismatch | Answer Scale Shift |
| `all` | Weighted average of the above | — |

**Stepwise**: multi-step samples are split into "predict the next two steps" atomic samples (30K originals → 69,937 atomic).

## Reproduction

- Base models: Llama2-7B / Qwen2.5-3B
- Fine-tuning: LoRA (rank=16, alpha=16), 1 epoch, lr=2e-5, max_length=512
- Training samples: 30,000 per experiment (stepwise: 30K originals split)
- Evaluation: GSM8K test (1,319), vLLM (temperature=0.0, top_p=1.0)
- Answer extraction: "The answer is:"

## Differences from the Paper

| Item | Paper | Code / actual |
|------|-------|---------------|
| Qwen model | "Qwen3-4B" | **Qwen2.5-3B** (paper to be corrected) |
| Training samples | "30,000" | **30,000** (consistent) |
| Main CoT training | Table 9 reports 30K (Llama 39.14%+-0.99) | a 100K "original T-CoT" reproduction exists (Llama 45.03%, not reported) |

## Results (camera-ready)

All numbers are means over three seeds (42, 2026, 12345); +- denotes the standard deviation reported in the paper.

Table 9 - baseline and CoT vs. No-CoT:

| Model | Experiment | Acc (%) |
|---|---|---|
| Llama2-7B | Untrained | 2.50 |
| Llama2-7B | CoT training | 39.14+-0.99 |
| Llama2-7B | No-CoT training | 8.16+-2.50 |
| Qwen2.5-3B | Untrained | 64.44 |
| Qwen2.5-3B | CoT training | 79.23+-0.25 |
| Qwen2.5-3B | No-CoT training | 19.41+-1.17 |

Table 10 - distribution shift:

| Model | Shift type | Acc (%) |
|---|---|---|
| Llama2-7B | Scale | 38.74+-0.70 |
| Llama2-7B | Complexity | 36.57+-0.17 |
| Llama2-7B | Topic | 37.51+-2.14 |
| Qwen2.5-3B | Topic | 78.60+-0.55 |
| Qwen2.5-3B | Complexity | 79.53+-0.45 |
| Qwen2.5-3B | Answer scale | 79.10+-0.25 |

Table 11 - training-time reasoning noise:

| Model | p=0 | p=0.05 | p=0.10 | p=0.20 | p=0.50 |
|---|---|---|---|---|---|
| Llama2-7B | 39.14+-0.99 | 39.20+-0.56 | 39.73+-1.02 | 37.63+-0.88 | 37.66+-0.89 |
| Qwen2.5-3B | 79.23+-0.25 | 79.27+-0.25 | 78.57+-0.37 | 77.20+-0.60 | 75.69+-0.56 |
