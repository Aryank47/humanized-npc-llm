# Fine-Tuning

**Owner**: Subodh Kant (M23CSA531)

## Setup
```bash
pip install torch transformers peft accelerate bitsandbytes
```

## Run Training
```bash
python train.py
```

## Test Inference
```bash
python inference.py --prompt "Can you repair my sword?"
```

## Outputs
- `outputs/model/` - LoRA adapters
- `outputs/logs/` - Training logs
