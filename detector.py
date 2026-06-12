"""
SafeSite AI — PPE Detector
Connects to RTSP camera stream, detects PPE violations, sends WhatsApp alerts.
"""

import cv2
import time
import csv
import os
import base64
import requests
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────────────
RTSP_URL        = os.getenv("RTSP_URL", "0")          # Use "0" for webcam while testing
MODEL_PATH      = os.getenv("MODEL_PATH", "yolov8n.pt")  # Swap with best.pt after training
SITE_NAME       = os.getenv("SITE_NAME", "Site A - Whitefield")
SUPERVISOR_PHONE= os.getenv("SUPERVISOR_PHONE", "")   # e.g. 9876543210
ALERT_COOLDOWN  = int(os.getenv("ALERT_COOLDOWN", "60"))  # seconds between alerts
CONFIDENCE      = float(os.getenv("CONFIDENCE", "0.5"))

VIOLATIONS_DIR  = "violations"
LOG_FILE        = "logs/violations.csv"
os.makedirs(VIOLATIONS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ── VIOLATION RULES ──────────────────────────────────────────────────────────
# These are the class names your trained model will detect.
# Default yolov8n detects "person" — swap these after training on PPE dataset.
VIOLATION_CLASSES = {
    "no-helmet":    {"label": "No Helmet ⛑️",   "severity": "HIGH"},
    "no-vest":      {"label": "No Safety Vest", "severity": "MEDIUM"},
    "no-gloves":    {"label": "No Gloves",      "severity": "LOW"},
}

# For testing with base yolov8 (before PPE training), flag all "person" detections
TEST_MODE = os.getenv("TEST_MODE", "true").lower() == "true"


# ── WHATSAPP ALERT (Twilio) ───────────────────────────────────────────────────
def send_whatsapp_alert(frame, violations: list):
    """Send WhatsApp alert with screenshot to supervisor."""
    twilio_sid   = os.getenv("TWILIO_SID", "")
    twilio_token = os.getenv("TWILIO_TOKEN", "")
    from_number  = os.getenv("TWILIO_FROM", "whatsapp:+14155238886")

    if not twilio_sid or not SUPERVISOR_PHONE:
        print("⚠️  Twilio not configured — skipping WhatsApp alert")
        return

    try:
        from twilio.rest import Client
        client = Client(twilio_sid, twilio_token)

        violation_lines = "\n".join([
            f"• {v['label']} [{v['severity']}]" for v in violations
        ])

        body = (
            f"🚨 *SAFETY VIOLATION DETECTED*\n"
            f"📍 Site: {SITE_NAME}\n"
            f"🕐 Time: {datetime.now().strftime('%d %b %Y, %I:%M %p')}\n\n"
            f"*Violations:*\n{violation_lines}\n\n"
            f"Please take immediate action."
        )

        client.messages.create(
            body=body,
            from_=from_number,
            to=f"whatsapp:+91{SUPERVISOR_PHONE}"
        )
        print(f"✅ WhatsApp alert sent to {SUPERVISOR_PHONE}")

    except Exception as e:
        print(f"❌ WhatsApp alert failed: {e}")


# ── SAVE VIOLATION ────────────────────────────────────────────────────────────
def save_violation(frame, violations: list) -> str:
    """Save screenshot and log to CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path  = f"{VIOLATIONS_DIR}/{timestamp}.jpg"
    cv2.imwrite(img_path, frame)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        for v in violations:
            writer.writerow([
                timestamp,
                SITE_NAME,
                v["label"],
                v["severity"],
                img_path
            ])

    print(f"💾 Saved: {img_path}")
    return img_path


# ── DETECTION LOGIC ───────────────────────────────────────────────────────────
def check_violations(results, model) -> list:
    """Extract violations from YOLO results."""
    violations = []

    for box in results[0].boxes:
        class_id   = int(box.cls)
        label      = model.names[class_id]
        confidence = float(box.conf)

        if confidence < CONFIDENCE:
            continue

        # After PPE training — check for specific violation classes
        if label in VIOLATION_CLASSES:
            violations.append({
                "label":    VIOLATION_CLASSES[label]["label"],
                "severity": VIOLATION_CLASSES[label]["severity"],
                "raw":      label
            })

        # TEST MODE — flag any person detected (to test before PPE model is trained)
        elif TEST_MODE and label == "person":
            violations.append({
                "label":    "Person Detected (Test Mode)",
                "severity": "INFO",
                "raw":      label
            })

    return violations


# ── CONNECT TO CAMERA ─────────────────────────────────────────────────────────
def connect_camera(url):
    """Connect to camera with retry logic."""
    # Convert "0" to integer for webcam
    source = int(url) if url.isdigit() else url
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"❌ Cannot connect to camera: {url}")
        return None

    print(f"✅ Camera connected: {url}")
    return cap


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
def run():
    print("=" * 50)
    print("🏗️  SafeSite AI — Starting up")
    print(f"📍 Site: {SITE_NAME}")
    print(f"📷 Camera: {RTSP_URL}")
    print(f"🤖 Model: {MODEL_PATH}")
    print(f"🧪 Test Mode: {TEST_MODE}")
    print("=" * 50)

    # Initialise CSV log with headers if new file
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "site", "violation", "severity", "screenshot"])

    # Load model
    print("⏳ Loading AI model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded")

    # Connect camera
    cap = connect_camera(RTSP_URL)
    if cap is None:
        return

    last_alert_time = 0
    frame_count     = 0

    print("\n🔍 Monitoring started. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()

        # Auto-reconnect if camera drops
        if not ret:
            print("⚠️  Camera feed lost — reconnecting in 5s...")
            time.sleep(5)
            cap = connect_camera(RTSP_URL)
            if cap is None:
                break
            continue

        frame_count += 1

        # Only analyse every 5th frame (saves CPU, still catches violations)
        if frame_count % 5 != 0:
            cv2.imshow("SafeSite AI Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Run AI detection
        results    = model(frame, verbose=False)
        violations = check_violations(results, model)

        # Draw bounding boxes on frame
        annotated = results[0].plot()

        # Add site name and timestamp to frame
        timestamp_text = datetime.now().strftime("%d %b %Y  %I:%M:%S %p")
        cv2.putText(annotated, f"SafeSite AI | {SITE_NAME}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, timestamp_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if violations:
            now = time.time()

            # Red alert banner on frame
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 80), (0, 0, 200), -1)
            cv2.putText(annotated, "⚠ VIOLATION DETECTED",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            print(f"\n🚨 Violations detected at {datetime.now().strftime('%H:%M:%S')}:")
            for v in violations:
                print(f"   • {v['label']} [{v['severity']}]")

            # Save screenshot every time
            save_violation(annotated, violations)

            # Send alert with cooldown (max 1 per minute)
            if now - last_alert_time > ALERT_COOLDOWN:
                send_whatsapp_alert(annotated, violations)
                last_alert_time = now
            else:
                remaining = int(ALERT_COOLDOWN - (now - last_alert_time))
                print(f"   (Next alert in {remaining}s)")

        # Show live feed
        cv2.imshow("SafeSite AI Monitor", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ SafeSite AI stopped.")


if __name__ == "__main__":
    run()
