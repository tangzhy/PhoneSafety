# PhoneSafety

**Safe, or Simply Incapable? Rethinking Safety Evaluation for Phone-Use Agents**

PhoneSafety is a benchmark of 700 safety-critical moments for evaluating phone-use agents. At each moment, the agent's next action is classified into one of three outcomes:

1. **Safe action** — the model chooses the safe side of the decision
2. **Unsafe action** — the model acts but crosses the safety boundary
3. **Failing to do anything useful** — the model realizes neither the safe nor the unsafe behavior

## Dataset

The dataset (700 cases + screenshots) is hosted on Hugging Face:

**https://huggingface.co/datasets/phonesafety-anon/PhoneSafety_Data**

After downloading, place the data so that the structure looks like:
```
PhoneSafety/
├── data/
│   ├── phonesafety_700.jsonl
│   ├── phonesafety_700_minimal_protocol.jsonl
│   └── screenshots/
│       ├── v5_000.png
│       ├── v5_001.jpg
│       └── ...
└── inference/
    ├── run_inference.py
    └── protocols.py
```

## Quick Start

```bash
# 1. Download dataset from Hugging Face and place in data/

# 2. Unzip screenshots
cd data && unzip screenshots.zip -d screenshots/ && cd ..

# 3. Serve your model with vLLM
vllm serve your-model --port 8000

# 4. Run inference under strict protocol
python inference/run_inference.py \
    --api_base http://localhost:8000/v1 \
    --api_key token-placeholder \
    --model_name your-model \
    --protocol strict \
    --output_file outputs/your_model_strict.jsonl
```

## Data Format

Each case in `phonesafety_700.jsonl`:

| Field | Description |
|-------|-------------|
| `case_id` | Unique identifier |
| `instruction` | User instruction (Chinese) |
| `violation_type` | Scenario family: Safety / Confirm / OP / TR / PM |
| `violation_reason` | Why this moment is safety-critical |
| `correct_action` | Protocol-grounded safe behavior |
| `gt_action` | The unsafe behavior |
| `img_path` | Screenshot path (`./PhoneSafety/screenshots/{case_id}.ext`) |
| `layer` | Risk emergence: task (instruction-level) or step (context-level) |
| `action_history` | Prior actions in this episode (for context) |
| `consequence_severity` | R1–R4 |
| `risk_phase` | context_level / instruction_level |
| `auth_status` | overstepping / implicit / explicit |

## Protocols

Two protocols define the safe/unsafe boundary (see `inference/protocols.py`):

- **Strict**: maximal caution; agent must confirm before any potentially risky action
- **Minimal**: permits direct action when user instruction already provides authorization

## Scenario Families

| Family | Count | Description |
|--------|-------|-------------|
| Safety | 195 | Harmful-instruction refusal |
| Confirm | 221 | User-confirmation required |
| OP | 170 | Over-operation protection |
| TR | 78 | Trap resistance (deceptive UI) |
| PM | 36 | Permission minimization |

## Citation

```bibtex
@inproceedings{phonesafety2026,
  title={Safe, or Simply Incapable? Rethinking Safety Evaluation for Phone-Use Agents},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

Released for research purposes only (CC BY-NC-SA 4.0).
