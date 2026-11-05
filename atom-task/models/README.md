# Models

Place the base model weights (Qwen3-4B) here:

```
models/
└── qwen3-4b/          # e.g. Qwen3-4B from HuggingFace
    ├── config.json
    ├── model-*.safetensors
    └── tokenizer.json
```

All experiment scripts default to `${ROOT_DIR}/models/qwen3-4b`; override with `MODEL_PATH`:

```bash
export MODEL_PATH=/path/to/qwen3-4b
CUDA_VISIBLE_DEVICES=0 bash run_scaling.sh all_3step full
```

> Model weights are large and git-ignored.
