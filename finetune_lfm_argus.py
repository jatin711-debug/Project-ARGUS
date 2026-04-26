"""
===========================================================================
 PROJECT ARGUS — LFM2.5-VL Fine-Tuning for Orbital Military Detection
===========================================================================
 Run this in Google Colab (T4 GPU or better).
 
 What it does:
   1. Installs dependencies (Unsloth, TRL, etc.)
   2. Downloads MVRSD military vehicle dataset from GitHub
   3. Converts bbox annotations → VLM conversation pairs
   4. Fine-tunes LFM2.5-VL-450M with LoRA (4-bit QLoRA)
   5. Saves the adapter to Google Drive + pushes to HuggingFace Hub
 
 Usage:
   - Open Google Colab, set runtime to GPU (T4 free tier works)
   - Paste this entire script into a cell and run
   - Or split at the "# ── CELL" markers into separate cells
===========================================================================
"""

# ── CELL 1: Install Dependencies ─────────────────────────────────────────
# fmt: off
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("unsloth")
install("trl>=0.15")
install("datasets")
install("bitsandbytes")
install("accelerate")
install("peft")
install("Pillow")
# fmt: on

print("All dependencies installed.")

# ── CELL 2: Configuration ────────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class Config:
    # Model
    model_name: str = "LiquidAI/LFM2.5-VL-450M"
    max_seq_length: int = 2048
    load_in_4bit: bool = True            # QLoRA — fits on T4 16GB
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    
    # Training
    num_epochs: int = 3
    batch_size: int = 2
    grad_accum_steps: int = 4            # effective batch = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    max_train_samples: int = 4000        # ~4.8 GB RAM — safe for 12.7 GB Colab
    max_dota_samples: int = 3500         # stream 3500 DOTA images max
    
    # Dataset
    dataset_repo: str = "https://github.com/baidongls/MVRSD.git"
    dataset_dir: str = "/content/MVRSD"
    image_size: int = 640                # MVRSD native size
    
    # Output
    output_dir: str = "/content/argus-lfm-lora"
    hub_model_id: str = ""               # set to push to HF Hub

cfg = Config()
print(f"Config ready: {cfg.model_name}, LoRA r={cfg.lora_r}, epochs={cfg.num_epochs}")

# ── CELL 3: Download & Parse Dataset ─────────────────────────────────────

import os, glob, json, random, zipfile, shutil
from pathlib import Path
from PIL import Image

# ── Step 1: Get MVRSD demo data ──────────────────────────────────────────
# The full MVRSD dataset is hosted on Baidu Cloud Drive.
# The GitHub repo contains a demo.zip with sample images + labels.
# We extract that, then supplement with HuggingFace DOTA data.

MVRSD_DIR = cfg.dataset_dir

if not os.path.exists(MVRSD_DIR):
    print("Cloning MVRSD repository...")
    os.system(f"git clone --depth 1 {cfg.dataset_repo} {MVRSD_DIR}")

# Extract demo.zip if it exists
demo_zip = os.path.join(MVRSD_DIR, "demo.zip")
demo_extracted = os.path.join(MVRSD_DIR, "demo")
if os.path.exists(demo_zip) and not os.path.exists(demo_extracted):
    print(f"Extracting demo.zip...")
    with zipfile.ZipFile(demo_zip, "r") as z:
        z.extractall(MVRSD_DIR)
    print("Extracted.")

# ── Step 2: Also pull DOTA from HuggingFace ──────────────────────────────
# HichTala/dota loads natively without custom scripts.
DOTA_DIR = "/content/dota_data"
USE_HF_DOTA = True
hf_dota = None

if USE_HF_DOTA:
    print("Loading DOTA satellite dataset from HuggingFace (streaming)...")
    try:
        from datasets import load_dataset
        # Streaming = no full download, images loaded on-the-fly
        hf_dota = load_dataset("HichTala/dota", split="train", streaming=True)
        os.makedirs(DOTA_DIR, exist_ok=True)
        print(f"  HF DOTA stream ready (will cap at {cfg.max_dota_samples} samples)")
    except Exception as e:
        print(f"  HF DOTA download failed ({e}), using MVRSD only.")
        USE_HF_DOTA = False
        hf_dota = None

# ── Step 3: Class mappings ───────────────────────────────────────────────

MVRSD_CLASSES = {
    0: "Small Military Vehicle",
    1: "Large Military Vehicle",
    2: "Armored Fighting Vehicle",
    3: "Military Construction Vehicle",
    4: "Civilian Vehicle",
}

THREAT_BY_CLASS = {
    0: "MEDIUM",   # SMV
    1: "HIGH",     # LMV
    2: "HIGH",     # AFV
    3: "MEDIUM",   # MCV
    4: "LOW",      # CV
}

REASONING_TEMPLATES = [
    "{label} identified on unpaved road, possible forward deployment",
    "{label} detected near tree cover, likely concealed staging area",
    "{label} observed in open terrain, high visibility exposure",
    "{label} positioned near infrastructure, possible logistics hub",
    "{label} detected in convoy formation, indicates active movement",
    "{label} visible in desert terrain, limited concealment",
    "{label} spotted near airstrip perimeter, possible base security",
    "{label} located in urban area, mixed civilian-military zone",
]

def parse_yolo_annotation(txt_path: str, img_w: int, img_h: int):
    """Parse YOLO-format annotation file -> list of detection dicts."""
    detections = []
    if not os.path.exists(txt_path):
        return detections
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = [float(x) for x in parts[1:5]]
            x1 = round(max(0.0, cx - w / 2), 4)
            y1 = round(max(0.0, cy - h / 2), 4)
            x2 = round(min(1.0, cx + w / 2), 4)
            y2 = round(min(1.0, cy + h / 2), 4)

            label = MVRSD_CLASSES.get(cls_id, "Military Vehicle")
            threat = THREAT_BY_CLASS.get(cls_id, "MEDIUM")
            reasoning = random.choice(REASONING_TEMPLATES).format(label=label)

            detections.append({
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "threat_level": threat,
                "confidence": round(random.uniform(0.80, 0.99), 2),
                "reasoning": reasoning,
            })
    return detections


# ── Step 4: Detection prompt (same one used in ARGUS pipeline) ───────────

DETECTION_PROMPT = (
    "You are an orbital intelligence analyst examining satellite imagery "
    "from a defense reconnaissance satellite at ~800 km altitude.\n\n"
    "Detect ALL military-relevant objects visible in this image. "
    "For each object, provide:\n"
    '- "label": specific type of object\n'
    '- "bbox": normalized bounding box [x1, y1, x2, y2] in [0,1]\n'
    '- "threat_level": "LOW", "MEDIUM", or "HIGH"\n'
    '- "confidence": 0.0 to 1.0\n'
    '- "reasoning": brief tactical assessment\n\n'
    'Return a JSON array. If no targets visible, return: []'
)


# ── Step 5: Build conversation pairs ─────────────────────────────────────

def find_image_label_pairs(root_dir: str):
    """Recursively find all (image, label.txt) pairs under root_dir."""
    pairs = []
    all_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        all_images.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))

    for img_path in all_images:
        stem = Path(img_path).stem
        parent = Path(img_path).parent
        # Try sibling labels/ folder
        lbl_folder = parent.parent / "labels"
        txt_path = lbl_folder / f"{stem}.txt"
        if not txt_path.exists():
            # Try same folder
            txt_path = parent / f"{stem}.txt"
        pairs.append((str(img_path), str(txt_path)))

    return pairs


# DOTA class → military label mapping (for HF DOTA dataset)
DOTA_MILITARY_MAP = {
    "plane": ("Military Aircraft", "HIGH"),
    "ship": ("Naval Vessel", "HIGH"),
    "helicopter": ("Helicopter", "HIGH"),
    "large-vehicle": ("Large Military Vehicle", "MEDIUM"),
    "small-vehicle": ("Small Military Vehicle", "LOW"),
    "harbor": ("Harbor Installation", "MEDIUM"),
    "bridge": ("Strategic Bridge", "MEDIUM"),
    "storage-tank": ("Storage Tank", "MEDIUM"),
    "container-crane": ("Port Crane", "LOW"),
    "airport": ("Airfield", "HIGH"),
    "helipad": ("Helipad", "MEDIUM"),
}

# DOTA integer class IDs → string names (standard DOTA ordering)
DOTA_ID_TO_NAME = {
    0: "plane", 1: "ship", 2: "storage-tank", 3: "baseball-diamond",
    4: "tennis-court", 5: "basketball-court", 6: "ground-track-field",
    7: "harbor", 8: "bridge", 9: "large-vehicle", 10: "small-vehicle",
    11: "helicopter", 12: "roundabout", 13: "soccer-ball-field",
    14: "swimming-pool", 15: "container-crane", 16: "airport", 17: "helipad",
}


def build_dataset_entries():
    """Build VLM training conversation pairs from MVRSD + HF DOTA."""
    entries = []

    # ── Source 1: MVRSD (demo.zip extracted images) ──────────────────
    pairs = find_image_label_pairs(MVRSD_DIR)
    print(f"  Found {len(pairs)} image files in MVRSD directory")

    for img_path, txt_path in pairs:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        detections = parse_yolo_annotation(txt_path, img.width, img.height)
        response_json = json.dumps(detections, separators=(",", ":"))

        entries.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": DETECTION_PROMPT},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": response_json},
                ]},
            ],
        })

    print(f"  MVRSD entries: {len(entries)}")

    # ── Source 2: HuggingFace DOTA dataset (streamed, capped) ────────
    if hf_dota is not None:
        dota_count = 0
        debug_printed = False
        for row in hf_dota:
            if dota_count >= cfg.max_dota_samples:
                break

            # Debug: print first row structure so we know the format
            if not debug_printed:
                print(f"  [DEBUG] DOTA row keys: {list(row.keys())}")
                for k, v in row.items():
                    if k != "image":
                        print(f"    {k}: {type(v).__name__} = {str(v)[:200]}")
                debug_printed = True

            try:
                img = row.get("image")
                if img is None:
                    continue
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                else:
                    continue

                w, h = img.size
                detections = []

                # Try multiple annotation formats
                objects = row.get("objects", row.get("annotations", None))

                if isinstance(objects, dict):
                    # Format: {"category": [...], "bbox": [[...], ...]}
                    cats = objects.get("category", objects.get("categories",
                           objects.get("label", objects.get("labels", []))))
                    bboxes = objects.get("bbox", objects.get("bboxes",
                             objects.get("bounding_box", [])))

                    for j, cat in enumerate(cats):
                        # Resolve category to string
                        if isinstance(cat, int):
                            cat_str = DOTA_ID_TO_NAME.get(cat, None)
                        else:
                            cat_str = str(cat).lower().strip()

                        if cat_str not in DOTA_MILITARY_MAP:
                            continue
                        label, threat = DOTA_MILITARY_MAP[cat_str]

                        if j < len(bboxes):
                            b = bboxes[j]
                            if isinstance(b, dict):
                                b = [b.get("x1", b.get("xmin", 0)),
                                     b.get("y1", b.get("ymin", 0)),
                                     b.get("x2", b.get("xmax", 0)),
                                     b.get("y2", b.get("ymax", 0))]
                            if max(b) > 1.0:
                                x1 = round(b[0] / w, 4)
                                y1 = round(b[1] / h, 4)
                                x2 = round(b[2] / w, 4)
                                y2 = round(b[3] / h, 4)
                            else:
                                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                        else:
                            continue

                        reasoning = random.choice(REASONING_TEMPLATES).format(label=label)
                        detections.append({
                            "label": label,
                            "bbox": [x1, y1, x2, y2],
                            "threat_level": threat,
                            "confidence": round(random.uniform(0.80, 0.99), 2),
                            "reasoning": reasoning,
                        })

                elif isinstance(objects, list):
                    # Format: [{"category": "plane", "bbox": [...]}, ...]
                    for obj in objects:
                        cat = obj.get("category", obj.get("label", ""))
                        cat_str = str(cat).lower().strip() if isinstance(cat, str) else DOTA_ID_TO_NAME.get(cat, "")
                        if cat_str not in DOTA_MILITARY_MAP:
                            continue
                        label, threat = DOTA_MILITARY_MAP[cat_str]
                        b = obj.get("bbox", obj.get("bounding_box", [0,0,0,0]))
                        if max(b) > 1.0:
                            x1, y1 = round(b[0]/w, 4), round(b[1]/h, 4)
                            x2, y2 = round(b[2]/w, 4), round(b[3]/h, 4)
                        else:
                            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

                        reasoning = random.choice(REASONING_TEMPLATES).format(label=label)
                        detections.append({
                            "label": label,
                            "bbox": [x1, y1, x2, y2],
                            "threat_level": threat,
                            "confidence": round(random.uniform(0.80, 0.99), 2),
                            "reasoning": reasoning,
                        })

                response_json = json.dumps(detections, separators=(",", ":"))
                entries.append({
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": DETECTION_PROMPT},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": response_json},
                        ]},
                    ],
                })
                dota_count += 1
            except Exception as ex:
                if not debug_printed:
                    print(f"  [DEBUG] DOTA parse error: {ex}")
                continue

        print(f"  DOTA entries: {dota_count}")

    total_with_targets = sum(
        1 for e in entries if '"label"' in e["messages"][1]["content"][0]["text"]
    )
    print(f"  Total: {len(entries)} ({total_with_targets} with targets, {len(entries) - total_with_targets} empty)")

    random.shuffle(entries)
    if cfg.max_train_samples and len(entries) > cfg.max_train_samples:
        entries = entries[:cfg.max_train_samples]
    return entries


print("Building training dataset...")
dataset_entries = build_dataset_entries()
print(f"Dataset ready: {len(dataset_entries)} training samples")

if len(dataset_entries) > 0:
    sample_resp = dataset_entries[0]["messages"][1]["content"][0]["text"]
    print(f"Sample response: {sample_resp[:200]}...")

# ── CELL 4: Load Model with Unsloth ─────────────────────────────────────

from unsloth import FastVisionModel

print(f"Loading {cfg.model_name} with 4-bit quantization...")
model, processor = FastVisionModel.from_pretrained(
    model_name=cfg.model_name,
    max_seq_length=cfg.max_seq_length,
    load_in_4bit=cfg.load_in_4bit,
)

print("Applying LoRA adapters...")
model = FastVisionModel.get_peft_model(
    model,
    r=cfg.lora_r,
    target_modules=list(cfg.lora_target_modules),
    lora_alpha=cfg.lora_alpha,
    use_gradient_checkpointing="unsloth",  # 2x faster
)

# Print trainable parameter count
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.2f}%)")

# SFTTrainer requires a HF Dataset — the PIL image serialization takes
# ~8 min for 3500 images. This is a ONE-TIME cost before training.
from datasets import Dataset

print("Converting to HF Dataset (encoding images — takes ~8 min, be patient)...")
train_dataset = Dataset.from_list(dataset_entries)
print(f"HF Dataset ready: {len(train_dataset)} rows")

# ── CELL 6: Train ───────────────────────────────────────────────────────

from trl import SFTTrainer, SFTConfig
from unsloth import UnslothVisionDataCollator

training_args = SFTConfig(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_epochs,
    per_device_train_batch_size=cfg.batch_size,
    gradient_accumulation_steps=cfg.grad_accum_steps,
    learning_rate=cfg.learning_rate,
    warmup_steps=cfg.warmup_steps,
    bf16=True,
    fp16=False,
    logging_steps=10,
    save_steps=250,
    save_total_limit=2,
    seed=42,
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    dataloader_pin_memory=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=UnslothVisionDataCollator(model, processor),
    processing_class=processor,
)

print("Starting training...")
print(f"  Epochs: {cfg.num_epochs}")
print(f"  Batch size: {cfg.batch_size} x {cfg.grad_accum_steps} accum = {cfg.batch_size * cfg.grad_accum_steps} effective")
print(f"  Learning rate: {cfg.learning_rate}")

stats = trainer.train()

print(f"\nTraining complete!")
print(f"  Total steps: {stats.global_step}")
print(f"  Final loss: {stats.training_loss:.4f}")

# ── CELL 7: Test Inference ───────────────────────────────────────────────

import torch

FastVisionModel.for_inference(model)

# Pick a random sample from training data for quick sanity check
test_entry = random.choice(dataset_entries)
test_image = test_entry["messages"][0]["content"][0]["image"]

conversation = [
    {"role": "user", "content": [
        {"type": "image", "image": test_image},
        {"type": "text", "text": DETECTION_PROMPT},
    ]},
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        min_p=0.15,
        repetition_penalty=1.05,
    )

# Decode ONLY the new tokens (strip the input prompt)
input_len = inputs["input_ids"].shape[1]
new_tokens = outputs[:, input_len:]
result = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

print("=" * 60)
print("TEST INFERENCE")
print("=" * 60)
print(f"Ground truth: {test_entry['messages'][1]['content'][0]['text'][:300]}")
print(f"\nModel output: {result[:300]}")
print("=" * 60)

# ── CELL 8: Save Model ──────────────────────────────────────────────────

# Save locally
print(f"Saving LoRA adapter to {cfg.output_dir}...")
model.save_pretrained(cfg.output_dir)
processor.save_pretrained(cfg.output_dir)

# Save to Google Drive (if mounted)
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    drive_path = "/content/drive/MyDrive/argus-lfm-lora"
    os.makedirs(drive_path, exist_ok=True)
    model.save_pretrained(drive_path)
    processor.save_pretrained(drive_path)
    print(f"Saved to Google Drive: {drive_path}")
except Exception:
    print("Google Drive not available, skipping Drive save.")

# Push to HuggingFace Hub (optional)
if cfg.hub_model_id:
    print(f"Pushing to HuggingFace Hub: {cfg.hub_model_id}")
    model.push_to_hub(cfg.hub_model_id, private=True)
    processor.push_to_hub(cfg.hub_model_id, private=True)
    print("Pushed to Hub successfully!")

print("\n" + "=" * 60)
print(" FINE-TUNING COMPLETE")
print("=" * 60)
print(f" Adapter saved to: {cfg.output_dir}")
print(f" To use in ARGUS pipeline:")
print(f"   1. Copy the adapter folder to your ARGUS machine")
print(f"   2. Update .env: ARGUS_VLM_MODEL={cfg.output_dir}")
print(f"   3. Or merge adapter into base model for deployment")
print("=" * 60)
