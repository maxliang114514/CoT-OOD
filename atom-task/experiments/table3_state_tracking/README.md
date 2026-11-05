# State-Tracking Ablation (Effect of State Tracking in CoT)

## Paper Reference

- **Table 3, Figure 2**: Necessity of explicit state tracking in CoT.

## Design

The standard CoT format records the full `Task_Chain` and `Input_Sequence` at every step. This experiment masks that state information during training so the intermediate state `z_{i-1}` no longer carries complete information:

- **Standard format** (full state):
  ```
  Step 1: Task_Chain: [Duplicate, Reverse, Substitute] | Input_Sequence: [xxx] | → Output_Sequence: [xxx] | Task_Chain: [Reverse, Substitute]
  ...
  ```
- **Ablated format** (mask Task_Chain and Input_Sequence with `......`):
  ```
  Step 1: [Duplicate] Task_Chain: [......, ......] | Input_Sequence: [......] | → Output_Sequence: [xxx] | Task_Chain: [......, ......]
  ...
  ```

The `......` placeholders keep the format length close to the original, removing format length as a confound so any performance drop can be attributed to missing state information.

## Layout

```
state_tracking/
├── README.md
├── prompts/
│   └── prompt_cot_atom_ablation.txt  # Ablation prompt (example uses masked format)
├── results/                  # Results
└── scripts/
    ├── run_pipeline.sh       # Main script (5 ablation_no_* experiments)
    └── predict.sh            # Single-model vLLM inference
```

## Usage

```bash
cd scripts
CUDA_VISIBLE_DEVICES=0 bash run_pipeline.sh full
```

Datasets come from the shared `datasets/` (`ablation_no_*.jsonl`). Supports `train_only` / `test_only` / `full` (default).

## Output

Results go to `results/`. Masking state lowers 7-step accuracy markedly, confirming that explicit state tracking is essential for multi-step decomposition.

## Results (camera-ready)

All numbers are means over three seeds (42, 2026, 12345); +- denotes the standard deviation reported in the paper.

Table 3 - format ablation (explicit state tracking vs. not):

| Step | Reverse | Rotate | Duplicate | Deduplicate | Substitute | Avg. (Abl.) | Avg. (Orig.) |
|---|---|---|---|---|---|---|---|
| 3-step | 98.47+-1.05 | 95.53+-2.08 | 92.33+-4.97 | 96.93+-1.43 | 94.53+-4.20 | 95.56+-1.41 | 96.87+-0.23 |
| 5-step | 95.80+-2.69 | 88.13+-5.12 | 83.20+-6.95 | 91.73+-5.49 | 87.40+-7.08 | 89.25+-2.55 | 91.57+-0.34 |
| 7-step | 79.40+-10.16 | 67.80+-2.61 | 67.47+-6.13 | 62.47+-4.43 | 60.67+-20.02 | 67.56+-4.77 | 85.37+-0.61 |
