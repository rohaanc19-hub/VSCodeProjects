"""
SafeSite AI — PPE Model Training Script
Run this in Google Colab (free GPU) or locally with a GPU.

Step 1: pip install roboflow ultralytics
Step 2: Get free API key from roboflow.com
Step 3: Run this script
Step 4: Copy runs/detect/train/weights/best.pt → models/best.pt
Step 5: In .env, set MODEL_PATH=models/best.pt and TEST_MODE=false
"""

import os
from roboflow import Roboflow
from ultralytics import YOLO

# ── STEP 1: Download PPE Dataset from Roboflow ────────────────────────────────
# Sign up free at roboflow.com, create a workspace, get your API key
ROBOFLOW_API_KEY = "your_api_key_here"  # Replace with your key

print("Downloading PPE dataset from Roboflow...")
rf      = Roboflow(api_key=ROBOFLOW_API_KEY)

# This is a public PPE/hard hat dataset — free to use
project = rf.workspace("roboflow-universe-projects").project("hard-hat-sample")
dataset = project.version(2).download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")

# ── STEP 2: Train YOLOv8 on PPE Data ─────────────────────────────────────────
print("\nStarting training...")
model = YOLO("yolov8n.pt")  # Start from pretrained base

results = model.train(
    data  = f"{dataset.location}/data.yaml",
    epochs= 50,          # Increase to 100 for better accuracy
    imgsz = 640,
    batch = 16,
    name  = "safesite_ppe",
    patience = 10,       # Stop early if no improvement
    save  = True,
)

print("\n✅ Training complete!")
print(f"Best model saved at: runs/detect/safesite_ppe/weights/best.pt")
print("\nNext steps:")
print("1. Copy best.pt to your models/ folder")
print("2. In .env: set MODEL_PATH=models/best.pt")
print("3. In .env: set TEST_MODE=false")
print("4. Run: python detector.py")

# ── STEP 3: Validate Model Performance ───────────────────────────────────────
print("\nValidating model...")
metrics = model.val()
print(f"mAP50:    {metrics.box.map50:.3f}")   # Target > 0.8
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.p.mean():.3f}")
print(f"Recall:    {metrics.box.r.mean():.3f}")
