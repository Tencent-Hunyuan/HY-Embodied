# Training Data Format

`finetune.py` expects a JSONL file with one JSON object per line.
Each object must have a `"messages"` key containing a list of conversation turns.

## Text-only turn

```json
{
  "messages": [
    {"role": "user",      "content": "How do you open a fridge?"},
    {"role": "assistant", "content": "First, grasp the handle …"}
  ]
}
```

## Image + text turn

The `"image"` field accepts a local file path or a URL.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "text",  "text": "Describe the scene."}
      ]
    },
    {"role": "assistant", "content": "The image shows …"}
  ]
}
```

## Multi-turn conversation

```json
{
  "messages": [
    {"role": "user",      "content": "What objects are on the table?"},
    {"role": "assistant", "content": "There is a cup and a plate."},
    {"role": "user",      "content": "Which should the robot pick up first?"},
    {"role": "assistant", "content": "The cup, as it is lighter and closer to the edge."}
  ]
}
```

## Notes

- Only `"assistant"` turns contribute to the training loss; user and system turns are masked.
- Examples longer than `--max_seq_len` (default 2048) are silently skipped.
- `data/sample_train.jsonl` contains three example entries for quick smoke-testing.
