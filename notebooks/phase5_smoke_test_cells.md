# Phase 5 smoke test — Colab notebook cells

Copy each `## Cell N` code block below into its own cell in a new Colab notebook, in order. Runtime: **GPU** (Runtime → Change runtime type → T4 GPU).

Two files to upload when Cell 2 asks for them (already built and zipped locally):
- `data/training/smoke_test/mathwriting_subset.zip` (300 examples, ~7MB — Stage A)
- `data/training/smoke_test/consistency_slice.zip` (56 examples, ~26MB — Stage B)

Known gotchas baked into these cells already (found during Phases 1-4, not hypothetical):
- Qwen3.5's `AutoProcessor` returns a multimodal processor, not a plain tokenizer — passing it directly to `SFTTrainer` crashes. Cell 3 extracts `processor.tokenizer` explicitly.
- Chat template must apply with `enable_thinking=False` — otherwise the model learns to expect open-ended reasoning before answering, not the direct-output format everything else uses.
- Stage B's images were rendered at dpi=70 based on local vision-token math (dpi=150, the original default, would produce more tokens per 3-page window than the single image that already caused a confirmed CUDA OOM). Not yet empirically confirmed on real GPU hardware — if Cell 7/8 OOMs anyway, that's the first thing to turn down further. If it comfortably fits with room to spare, there may be room to go back and re-render at higher DPI for better legibility.
- Cell 8 must load Stage A's saved checkpoint, not the base model — that's the entire point of the two-stage design. Double-check the printed confirmation in Cell 8 before trusting the run.

---

## Cell 1: Install dependencies

```python
!pip install -q unsloth
!pip install -q trl peft accelerate bitsandbytes pillow
```

If you hit `AttributeError: module 'torch' has no attribute ...` right after this cell: restart the runtime (Runtime → Restart session) and re-run Cell 1 then Cell 3 fresh — this is almost always a stale-import issue from installing over an already-loaded torch, not a real incompatibility. Only if that doesn't fix it is the install command itself the problem.

---

## Cell 2: Upload and unzip the smoke-test data

```python
from google.colab import files
import zipfile, os

print("Upload mathwriting_subset.zip, then consistency_slice.zip (one at a time):")
uploaded = files.upload()

for fname in uploaded:
    with zipfile.ZipFile(fname, 'r') as z:
        z.extractall('.')
    print(f"Extracted {fname}")

print(os.listdir('.'))
```

---

## Cell 3: Load Qwen3.5-0.8B via Unsloth FastVisionModel + apply QLoRA

```python
from unsloth import FastVisionModel
import torch

model, processor = FastVisionModel.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

# Gotcha: processor is multimodal (image_processor + tokenizer bundled).
# SFTTrainer needs the plain text tokenizer, extracted explicitly.
tokenizer = processor.tokenizer

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,  # Unsloth convention: lora_alpha == r, not r*2
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("Model loaded and QLoRA applied.")
```

---

## Cell 4: Load and format the Stage A dataset (MathWriting)

```python
import json
from pathlib import Path
from PIL import Image

def load_jsonl_dataset(jsonl_path, base_dir):
    examples = []
    base_dir = Path(base_dir)
    for line in open(jsonl_path):
        ex = json.loads(line)
        ex["_image_path"] = str(base_dir / ex["image"])
        examples.append(ex)
    return examples

stage_a_data = load_jsonl_dataset("mathwriting_subset/examples.jsonl", "mathwriting_subset")
print(f"Stage A: {len(stage_a_data)} examples")
print(stage_a_data[0]["messages"][2]["content"][:100])
```

---

## Cell 5: Stage A training run

```python
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from PIL import Image

FastVisionModel.for_training(model)

def to_hf_format(ex):
    image = Image.open(ex["_image_path"]).convert("RGB")
    return {"messages": ex["messages"], "images": [image]}

stage_a_formatted = [to_hf_format(ex) for ex in stage_a_data]

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    train_dataset=stage_a_formatted,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,          # smoke test: a few dozen steps, not a full epoch
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="stage_a_checkpoint",
        report_to="none",
    ),
)

stage_a_result = trainer.train()
print(stage_a_result)

model.save_pretrained("stage_a_checkpoint")
tokenizer.save_pretrained("stage_a_checkpoint")
print("Stage A checkpoint saved to stage_a_checkpoint/")
```

---

## Cell 6: Stage A inference sanity check

```python
FastVisionModel.for_inference(model)

test_ex = stage_a_data[-1]  # a held-out example, not used in the training loop above's slice ordering
test_image = Image.open(test_ex["_image_path"]).convert("RGB")

messages = [
    {"role": "system", "content": test_ex["messages"][0]["content"]},
    {"role": "user", "content": [{"type": "text", "text": "Transcribe the handwritten math in this image into LaTeX."}, {"type": "image"}]},
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = processor(text=[prompt], images=[test_image], return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=200, use_cache=True)
print(processor.batch_decode(output, skip_special_tokens=True)[0])
print()
print("Expected (ground truth):", test_ex["messages"][2]["content"])
```

---

## Cell 7: Load and format the Stage B dataset (consistency slice, multi-page)

```python
import json

# NOT load_jsonl_dataset() here -- that helper expects MathWriting's format
# (a top-level "image" field pointing to an external file). The consistency
# slice has a different structure: images are embedded inline as base64
# inside messages[1]["content"], no top-level "image" key at all.
stage_b_data = [json.loads(line) for line in open("consistency_slice/examples.jsonl")]
print(f"Stage B: {len(stage_b_data)} examples")
print("n_pages in first example:", stage_b_data[0]["n_pages"])

def to_hf_format_multipage(ex):
    # user content alternates {"type":"text"} / {"type":"image_url"} blocks from
    # build_consistency_slice.py — pull out just the images, in order, for the
    # "images" list the collator expects; the text/image placeholders in
    # ex["messages"] already carry the right structure for apply_chat_template.
    image_urls = [b for b in ex["messages"][1]["content"] if b["type"] == "image_url"]
    images = []
    for b in image_urls:
        # these were embedded as base64 data URLs when the slice was built —
        # decode back into PIL images for the collator.
        import base64, io
        b64 = b["image_url"]["url"].split(",", 1)[1]
        images.append(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
    # Simplify the message content to plain {"type":"image"} placeholders,
    # matching what apply_chat_template / the collator expect at train time.
    simplified_content = []
    for b in ex["messages"][1]["content"]:
        if b["type"] == "image_url":
            simplified_content.append({"type": "image"})
        else:
            simplified_content.append(b)
    messages = [ex["messages"][0], {"role": "user", "content": simplified_content}, ex["messages"][2]]
    return {"messages": messages, "images": images}

stage_b_formatted = [to_hf_format_multipage(ex) for ex in stage_b_data]
print("First example image count:", len(stage_b_formatted[0]["images"]))
```

---

## Cell 8: Stage B training run — continues Stage A's checkpoint, not the base model

```python
from unsloth import FastVisionModel

# IMPORTANT: load Stage A's saved checkpoint here, NOT "Qwen/Qwen3.5-0.8B" again.
# This is the entire point of the two-stage design — verify the print below
# before trusting the rest of this run.
model_b, processor_b = FastVisionModel.from_pretrained(
    "stage_a_checkpoint",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
tokenizer_b = processor_b.tokenizer
print("Loaded from stage_a_checkpoint/ — confirm this path, not the base model name, before proceeding.")

FastVisionModel.for_training(model_b)

trainer_b = SFTTrainer(
    model=model_b,
    tokenizer=tokenizer_b,
    data_collator=UnslothVisionDataCollator(model_b, processor_b),
    train_dataset=stage_b_formatted,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_train_epochs=3,   # small dataset (56 examples) — repeated exposure is the point
        learning_rate=1e-4,   # lower than Stage A: fine-tuning an already-adapted checkpoint
        logging_steps=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="stage_b_checkpoint",
        report_to="none",
    ),
)

stage_b_result = trainer_b.train()
print(stage_b_result)

model_b.save_pretrained("stage_b_checkpoint")
tokenizer_b.save_pretrained("stage_b_checkpoint")
print("Stage B checkpoint saved to stage_b_checkpoint/")
```

---

## Cell 9: Stage B multi-image inference sanity check

```python
FastVisionModel.for_inference(model_b)

test_ex_b = stage_b_formatted[-1]
messages = [test_ex_b["messages"][0], test_ex_b["messages"][1]]
prompt = processor_b.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = processor_b(text=[prompt], images=test_ex_b["images"], return_tensors="pt").to("cuda")

output = model_b.generate(**inputs, max_new_tokens=500, use_cache=True)
print(processor_b.batch_decode(output, skip_special_tokens=True)[0])
```

---

## Cell 10: Zip and download both checkpoints

```python
from google.colab import files
import shutil

shutil.make_archive("stage_a_checkpoint", "zip", "stage_a_checkpoint")
shutil.make_archive("stage_b_checkpoint", "zip", "stage_b_checkpoint")

files.download("stage_a_checkpoint.zip")
files.download("stage_b_checkpoint.zip")
```

---

## What "passed" looks like

- No exceptions in any cell.
- Cell 5's logged loss trends downward over the 60 steps (sanity that learning is happening, not a quality bar).
- Cell 6 produces *some* LaTeX-shaped output (doesn't need to be correct yet).
- Cell 8 printed confirmation that it loaded from `stage_a_checkpoint/`, and its logged loss also trends downward.
- Cell 9 produces output referencing content from multiple pages, without crashing on the multi-image input.

If Cell 5 or Cell 8 OOMs: first response is to drop `per_device_train_batch_size` (already at the minimum of 1) — next lever is reducing `max_steps`/`num_train_epochs` is irrelevant to memory; the real fix is regenerating Stage B's images at a lower `--dpi` via `scripts/training_data/build_consistency_slice.py` (currently 70) and re-zipping, or reducing `--window-size` from 3 to 2 pages.
