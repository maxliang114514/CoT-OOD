# T-Scaling (Main Experiment) No-CoT Control

## Paper Reference

- **Table 1b**: No-CoT control for the main experiment (T-scaling).

The only difference from `t_scaling` is that training does **not** generate intermediate reasoning steps; the model directly outputs `Final Answer: [xxx]`. This shows that without intermediate steps the model cannot capture the compositional structure and performance degrades sharply.

## Layout

```
t_scaling_nocot/
├── README.md
├── datasets/                 # Reserved (shared datasets live in <root>/datasets/)
├── output/                   # Model output (git-ignored)
├── results/                  # Results
├── prompts/
│   └── prompt_nocot_atom.txt # No-CoT prompt
└── scripts/
    └── run_scaling.sh        # Main script (single_step / train_3step / train_mixed_3_5step)
```

## Usage

```bash
cd scripts

# Single-step
CUDA_VISIBLE_DEVICES=0 bash run_scaling.sh single_step full

# train_3step + train_mixed_3_5step (3 seeds)
CUDA_VISIBLE_DEVICES=0 bash run_scaling.sh all_3step full
```

Supports `train_only` / `test_only` / `full` (default).

## Output

Results go to `results/` (directory names carry a `_nocot` suffix); trained models to `output/`.

## Results (camera-ready)

All numbers are means over three seeds (42, 2026, 12345); +- denotes the standard deviation reported in the paper.

Table 1(b) - No-CoT setting:

| Training data | 3-step | 5-step | 7-step | 9-step |
|---|---|---|---|---|
| 1-step | 10.93+-0.84 | 7.00+-0.00 | 2.73+-0.25 | 0.67+-0.34 |
| 3-step | 75.87+-0.84 | 30.20+-0.65 | 11.00+-0.85 | 5.13+-0.09 |
| Mix3+5 | 51.00+-0.75 | 26.60+-1.40 | 12.20+-0.75 | 5.73+-0.09 |
