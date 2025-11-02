# Data Engineering

**Owner**: Aryan Kumar (M23CSA510)

## Setup
```bash
pip install -r ../requirements.txt
```

## Run
```bash
# Publishable mix (no Skyrim):
python -m data_engineering.cli \           
  --config data_engineering/config/mix.classwork.yaml \
  --out    data_engineering/data/processed/classwork
```

## Outputs
- `data/processed/publishable/train.jsonl`
- `data/processed/publishable/val.jsonl`
- `data/processed/publishable/test.jsonl`
- `data/processed/publishable/MANIFEST.json`

## For Team Members
Use the files in `data/` for training and evaluation.
Don't modify files in this folder!
