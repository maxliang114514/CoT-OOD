# T-Scaling (Main Experiment) + Operation Ablation (CoT)

## Paper Reference

- **Main experiment (T-scaling / reasoning depth)**: Figure 3a, Table 1a. Trains on `train_1step`, `train_3step`, `train_mixed_3_5step` and evaluates on 3/5/7/9-step test sets (short-to-long generalization).
- **Operation ablation (CoT)**: Table 2a. Trains on `ablation_no_*` datasets (one operation absent from multi-step chains but present in single-step data) to test compositional generalization.

## Layout

```
t_scaling/
├── README.md
├── datasets/
│   └── k_scaling/                 # k-scaling variants (CoT format length k1/k2/k3)
├── results/                       # Results
└── scripts/
    ├── run_scaling.sh             # Main experiment (single_step / train_3step / train_mixed_3_5step, 3 seeds)
    ├── run_ablation.sh            # Operation ablation (CoT, seed 42)
    ├── run_k_scaling.sh           # k-scaling (CoT format length)
    ├── generate_datasets.py       # Generate train_3step / train_mixed_3_5step
    ├── generate_9step_test.py     # Generate test_9step
    ├── generate_datasets_type9.py # Generate train_9step
    ├── generate_test_doubled.py   # Test-set expansion
    ├── generate_k_scaling_data.py # Generate k-scaling data
    └── predict.sh                 # Single-model vLLM inference
```

## Usage

```bash
# Main experiment (train_3step + train_mixed_3_5step, 3 seeds)
cd scripts
CUDA_VISIBLE_DEVICES=0 bash run_scaling.sh all_3step full

# Single-step experiment
CUDA_VISIBLE_DEVICES=0 bash run_scaling.sh single_step full

# Operation ablation (CoT)
CUDA_VISIBLE_DEVICES=0 bash run_ablation.sh full
```

`run_scaling.sh` supports `train_only` / `test_only` / `full` (default).

## Output

Results go to `results/`; trained models to `output/`. Each result directory contains `summary.txt`, `detailed_results.json`, and `test_<N>step.log`.

## Results (camera-ready)

All numbers are means over three seeds (42, 2026, 12345); +- denotes the standard deviation reported in the paper.

Table 1(a) - CoT setting, accuracy (%) by training data and test steps:

| Training data | 3-step | 5-step | 7-step | 9-step |
|---|---|---|---|---|
| 1-step | 91.80+-2.24 | 63.13+-4.15 | 36.27+-7.87 | 22.27+-3.67 |
| 3-step | 99.33+-0.25 | 95.73+-0.68 | 88.87+-1.54 | 79.27+-3.03 |
| Mix3+5 | 100.00+-0.00 | 97.87+-0.68 | 94.00+-0.59 | 90.40+-1.57 |

Table 2(a) - CoT, one operation excluded from multi-step training samples:

| Excluded | 3-step | 5-step | 7-step | 9-step |
|---|---|---|---|---|
| Reverse | 99.53+-0.34 | 97.13+-0.25 | 92.67+-1.60 | 87.13+-0.84 |
| Deduplicate | 97.87+-0.47 | 93.27+-1.11 | 86.20+-1.82 | 76.73+-0.66 |
| Substitute | 96.80+-0.43 | 90.73+-0.41 | 84.33+-0.38 | 70.67+-0.81 |
| Duplicate | 93.60+-0.59 | 84.80+-0.99 | 76.73+-1.75 | 66.73+-1.25 |
| Rotate | 96.53+-0.66 | 91.93+-0.66 | 86.93+-0.34 | 80.13+-0.41 |
