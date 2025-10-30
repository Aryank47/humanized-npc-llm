# Data Engineering

**Owner**: Aryan Kumar (M23CSA510)

## Setup
```bash
pip install datasets pandas jsonschema tqdm pyyaml requests selectolax python-slugify orjson
```

## Run
```bash
python run_pipeline.py
```

## Outputs
- `outputs/train.jsonl` (~55k examples)
- `outputs/val.jsonl` (~3k examples)
- `outputs/test.jsonl` (~3k examples)
- `outputs/MANIFEST.json`

## For Team Members
Use the files in `outputs/` for training and evaluation.
Don't modify files in this folder!
