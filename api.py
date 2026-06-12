"""
SafeSite AI — Backend API
Serves violation data and stats to the dashboard.
Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
from datetime import datetime, timedelta

app = FastAPI(title="SafeSite AI API")

# Allow dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve violation screenshots
if os.path.exists("violations"):
    app.mount("/screenshots", StaticFiles(directory="violations"), name="screenshots")

LOG_FILE = "logs/violations.csv"


def load_violations():
    """Load violations from CSV."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=["timestamp", "site", "violation", "severity", "screenshot"])
    df = pd.read_csv(LOG_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H%M%S", errors="coerce")
    return df


@app.get("/")
def root():
    return {"status": "SafeSite AI running", "time": datetime.now().isoformat()}


@app.get("/stats")
def get_stats():
    """Summary stats for dashboard."""
    df = load_violations()
    if df.empty:
        return {"today": 0, "week": 0, "total": 0, "high_severity": 0}

    now   = datetime.now()
    today = df[df["timestamp"].dt.date == now.date()]
    week  = df[df["timestamp"] >= now - timedelta(days=7)]
    high  = df[df["severity"] == "HIGH"]

    return {
        "today":         len(today),
        "week":          len(week),
        "total":         len(df),
        "high_severity": len(high)
    }


@app.get("/violations/recent")
def get_recent(limit: int = 20):
    """Most recent violations for live feed."""
    df = load_violations()
    if df.empty:
        return []

    recent = df.sort_values("timestamp", ascending=False).head(limit)
    records = []

    for _, row in recent.iterrows():
        screenshot_url = None
        if pd.notna(row.get("screenshot")):
            filename = os.path.basename(str(row["screenshot"]))
            screenshot_url = f"/screenshots/{filename}"

        records.append({
            "time":      row["timestamp"].strftime("%d %b, %I:%M %p") if pd.notna(row["timestamp"]) else "Unknown",
            "site":      row["site"],
            "violation": row["violation"],
            "severity":  row["severity"],
            "screenshot": screenshot_url
        })

    return records


@app.get("/violations/by-day")
def get_by_day():
    """Violations per day for bar chart (last 7 days)."""
    df = load_violations()
    if df.empty:
        return []

    now  = datetime.now()
    week = df[df["timestamp"] >= now - timedelta(days=7)].copy()

    if week.empty:
        return []

    week["date"]  = week["timestamp"].dt.date
    daily         = week.groupby("date").size().reset_index(name="count")
    daily["date"] = daily["date"].astype(str)

    return daily.to_dict(orient="records")


@app.get("/violations/by-type")
def get_by_type():
    """Violations broken down by type."""
    df = load_violations()
    if df.empty:
        return []

    by_type = df.groupby("violation").size().reset_index(name="count")
    return by_type.to_dict(orient="records")
