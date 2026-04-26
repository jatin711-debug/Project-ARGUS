# Project ARGUS — Orbital Intelligence Pipeline

**Hackathon:** Liquid AI x DPhi Space "AI in Space"  
**Track:** Liquid Track (LFM2-VL)  
**Domain:** Defense & Security Edge Compute

Multi-phase satellite imagery analysis pipeline that detects, validates, and
assesses targets from orbital feeds in real time. Decoy targets (2-D painted
fakes) are filtered automatically, and only a tiny JSON tactical report is
downlinked — bytes instead of gigabytes.

## Architecture — LFM-First Pipeline

```
Satellite API  →  Phase 1 (LFM2.5-VL)
                  Unified detection + localization + assessment
                  Single inference pass
                      ↓
                  Phase 2 (DA3 Depth)
                  3-D reality verification
                      ↓
               ┌──────┴──────┐
           3D REAL        2D DECOY
               │          (filtered)
               ↓              ↓
          Phase 3 (Report)  →  JSON downlink
```

| Phase | Model | Purpose |
|-------|-------|---------|
| 1 | **LFM2.5-VL-450M** | Unified detection, localization, classification & tactical assessment |
| 2 | Depth Anything 3 | 3-D reality check — filters flat decoys |
| 3 | — | Report assembly & downlink |

### Why LFM-First?

Traditional satellite intelligence pipelines chain 4-5 separate models
(detector → depth → segmenter → classifier). ARGUS puts Liquid AI's
LFM2.5-VL at the center — a single 450M-parameter model that performs
**detection, bounding-box localization, AND tactical reasoning** in one
inference pass. This approach:

- **Reduces VRAM** from 4 GPU-loaded models to 2
- **Eliminates class-locked detectors** — the VLM uses open-vocabulary
  prompting ("find military vehicles") instead of fixed DOTA classes
- **Runs on edge hardware** — the model runs on Jetson Orin, mobile devices,
  even in-browser via WebGPU

## Quick Start

### 1. Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Docker (for SimSat)
- Mapbox account (free tier)

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env to match your environment
```

> **Important — Mapbox token:** The `MAPBOX_ACCESS_TOKEN` env var must be set
> on the **SimSat Docker side** so SimSat can serve Mapbox imagery. Get a free
> token at [mapbox.com](https://www.mapbox.com/) and pass it when starting
> SimSat: `MAPBOX_ACCESS_TOKEN=pk.xxx docker compose up`

### 4. Run

```bash
python -m argus
```

## Configuration

All settings live in environment variables (loaded from `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ARGUS_API_URL` | `http://localhost:9005` | Satellite API base URL |
| `ARGUS_VLM_MODEL` | `LiquidAI/LFM2.5-VL-450M` | VLM model ID |
| `ARGUS_VLM_MAX_TOKENS` | `512` | Max tokens for VLM generation |
| `ARGUS_DA3_MODEL` | `depth-anything/DA3-BASE` | Depth model ID |
| `ARGUS_DEPTH_THRESH` | `0.1` | Depth std threshold for 3D check |
| `ARGUS_GHOST_STD` | `10.0` | Pixel std threshold for ghost images |
| `ARGUS_SCAN_INTERVAL` | `1` | Seconds between scans (0 = single shot) |

## Project Structure

```
argus/
├── __init__.py
├── __main__.py        # Entry point — scan loop
├── config.py          # Centralised settings + logger
├── loader.py          # Model registry (VLM + DA3)
├── models.py          # Target, DepthAnalysis dataclasses
├── pipeline.py        # 3-phase orchestrator
├── satellite.py       # SimSat API client
├── report.py          # JSON tactical report builder
└── phases/
    ├── __init__.py
    ├── detection.py   # Phase 1 — LFM2.5-VL detection + assessment
    └── depth.py       # Phase 2 — DA3 depth verification
```

## Key Features

- **LFM-first architecture** — LFM2.5-VL handles detection, localization,
  and tactical assessment in a single pass
- **Decoy filtering** — flat 2-D targets flagged by DA3 depth analysis
  are marked as decoys in the report
- **Ghost image detection** — ocean/blank Mapbox frames auto-skipped
- **Bandwidth savings** — report includes `edge_compute_savings` showing
  compression ratio vs. raw imagery (typically 10,000x+)
- **Edge-optimized** — 2 models totaling < 1B parameters

## Example Report

```json
{
  "mission": "PROJECT ARGUS",
  "satellite_position": { "lon": 35.9, "lat": 34.8, "alt": 793.1 },
  "targets_detected": 2,
  "decoys_filtered": 1,
  "targets": [
    {
      "class": "military vehicle convoy",
      "confidence": 0.85,
      "vlm_assessment": {
        "threat_level": "HIGH",
        "reasoning": "Three vehicles in formation on unpaved road..."
      },
      "depth_analysis": { "is_3d": true, "verdict": "REAL" }
    }
  ],
  "edge_compute_savings": {
    "compression_ratio": "29789x",
    "bandwidth_saved_pct": 99.97
  }
}
```
