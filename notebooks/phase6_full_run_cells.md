# Phase 6 — full two-stage training run

Same notebook structure as Phase 5's smoke test (`notebooks/phase5_smoke_test_cells.md`), which already passed cleanly on this exact GPU (Tesla T4, 14.56GB, no OOM at dpi=70). This file only calls out what's **different** for the full run — reuse Phase 5's cells for anything not mentioned below.

## What's changing from the smoke test

1. **Stage A gets the full dataset, not the 300-example subset.** Upload `data/training/mathwriting_slice.zip` (113MB, all 4,892 examples) instead of `mathwriting_subset.zip`. Stage B's data is unchanged — `consistency_slice.zip` already contained all 56 examples in the smoke test too (nothing to re-upload there, same file).
2. **Stage A trains for real epochs, not a fixed 60 steps.** With 4,892 examples, batch size 1, and gradient accumulation 4 (effective batch 4), one epoch is ~1,223 steps. Start with 1 epoch.
3. **Time expectation: hours, not minutes.** The smoke test's Stage B (42 steps) took ~9.5 minutes (~13.6s/step). Stage A's images are much smaller (288 vision tokens vs. Stage B's ~4,692/window) so its steps should be faster per-step, but there are ~29x more of them — expect somewhere in the ballpark of 2-4 hours for Stage A alone. **This is long enough to risk a Colab disconnect**, especially on the free tier (no background execution, session limits, occasional idle timeouts) — see the checkpointing change below, which exists specifically to survive that.
4. **Stage B epochs bumped modestly (3 → 5).** The smoke test's Stage B loss was still trending down at epoch 3 (1.63 → 0.33), not clearly plateaued, and 56 examples means extra epochs are cheap. Watch for the loss flattening out or reversing (overfitting on a small dataset) — if it does, that's a sign to stop increasing epochs further, not a bug.

## Cell 2 (upload) — only the filename changes

Upload `data/training/mathwriting_slice.zip` instead of `mathwriting_subset.zip` (same `consistency_slice.zip` as before). Everything else in Cell 2 stays the same — it already just extracts whatever zip you give it.

## Cell 4 — update the path

```python
stage_a_data = load_jsonl_dataset("mathwriting_slice/examples.jsonl", "mathwriting_slice")
print(f"Stage A: {len(stage_a_data)} examples")
```

(Only the folder name changed, from `mathwriting_subset` to `mathwriting_slice` — matches the unzipped full dataset's top-level folder.)

## Cell 5 — real training args, with checkpoint resilience

```python
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from PIL import Image
import os

FastVisionModel.for_training(model)

def to_hf_format(ex):
    image = Image.open(ex["_image_path"]).convert("RGB")
    return {"messages": ex["messages"], "images": [image]}

stage_a_formatted = [to_hf_format(ex) for ex in stage_a_data]

training_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    num_train_epochs=1,          # ~1223 steps over the full 4892 examples
    learning_rate=2e-4,
    logging_steps=20,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="stage_a_checkpoint",
    report_to="none",
    save_strategy="steps",
    save_steps=200,               # checkpoint every 200 steps -- survives a disconnect
    save_total_limit=3,           # keep only the last 3, so Colab's disk doesn't fill up
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    train_dataset=stage_a_formatted,
    args=training_args,
)

# If this session was disconnected and you're re-running after reconnecting,
# set RESUME = True below to continue from the last saved checkpoint instead
# of starting over from scratch.
RESUME = False
if RESUME and os.path.isdir("stage_a_checkpoint") and any(d.startswith("checkpoint-") for d in os.listdir("stage_a_checkpoint")):
    stage_a_result = trainer.train(resume_from_checkpoint=True)
else:
    stage_a_result = trainer.train()
print(stage_a_result)

model.save_pretrained("stage_a_checkpoint")
tokenizer.save_pretrained("stage_a_checkpoint")
print("Stage A checkpoint saved to stage_a_checkpoint/")
```

**If you get disconnected mid-run:** reconnect, re-run Cells 1-4 (reload everything), then in this cell set `RESUME = True` before running it again — it'll pick up from the last `save_steps` checkpoint instead of losing the progress.

## Cell 8 — same change (checkpointing) + bumped epochs

```python
from unsloth import FastVisionModel

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
        num_train_epochs=5,      # bumped from 3 -- smoke test loss was still trending down
        learning_rate=1e-4,
        logging_steps=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="stage_b_checkpoint",
        report_to="none",
        save_strategy="epoch",
        save_total_limit=2,
    ),
)

stage_b_result = trainer_b.train()
print(stage_b_result)

model_b.save_pretrained("stage_b_checkpoint")
tokenizer_b.save_pretrained("stage_b_checkpoint")
print("Stage B checkpoint saved to stage_b_checkpoint/")
```

## Everything else

Cells 1, 3, 6, 7, 9, 10 are unchanged from `phase5_smoke_test_cells.md` — same install, same model loading, same inference sanity checks, same download step. Just run the full notebook top to bottom with the Cell 2/4/5/8 changes above.

## If you have Colab Pro by the time you run this

Enable background execution (Runtime settings) so the training keeps running even if you close the browser tab — given the multi-hour estimate above, this matters a lot more here than it did for the few-minute smoke test.
