# 🔱 ARGUS — The AI That Never Forgets

> Real-time video AI agent with temporal memory, built on [Vision Agents SDK](https://github.com/GetStream/Vision-Agents) by Stream

[![Built for Vision Possible Hackathon](https://img.shields.io/badge/Vision%20Possible-Hackathon-00d4aa?style=flat-square)](https://www.wemakedevs.org/hackathons/vision)
[![Powered by Vision Agents](https://img.shields.io/badge/Powered%20by-Vision%20Agents-0052CC?style=flat-square)](https://github.com/GetStream/Vision-Agents)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square)](https://python.org)

---

<!-- ADD YOUR DEMO GIF HERE -->
<!-- ![ARGUS Demo](assets/demo.gif) -->

---

## The Problem

Every video AI today has amnesia.

It detects objects, responds, then immediately forgets. Ask it *"Where did I put my keys 5 minutes ago?"* — silence.

The Vision Agents SDK docs acknowledge this directly:
> *"Longer videos can cause the AI to lose context after ~30 seconds."*

**ARGUS solves this with a Temporal Memory Engine.**

```
Traditional Video AI:    Frame → Detect → Respond → FORGET
ARGUS:                   Frame → Detect → REMEMBER → Reason → Respond
```

---

## What ARGUS Can Do

| You Say | ARGUS Responds |
|---------|----------------|
| `"What do you see?"` | `"Person ID:2 at middle-center, Cup ID:3 at bottom-right"` |
| `"What am I holding?"` | `"You appear to be holding a bottle, ID:7"` |
| `"Where is the cup?"` | `"Cup ID:3 at bottom-right since 2:03 PM"` |
| `"What just moved?"` | `"Cup moved from bottom-left to bottom-right at 2:05 PM"` |
| `"Summarize everything"` | `"Person appeared 30s ago. Cup moved left to right at 2:05"` |

Response time: ~1 second. All answers come from memory, not re-analyzing video.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│               Vision Agents SDK                   │
│                                                  │
│  Camera Feed                                     │
│       │                                          │
│       ▼                                          │
│  YOLO26 Nano + ByteTrack                         │
│  (object detection + persistent IDs)             │
│       │                                          │
│       ▼                                          │
│  ┌────────────────────────────┐                  │
│  │   Temporal Memory Engine   │  ← core innovation│
│  │                            │                  │
│  │   Object Registry          │                  │
│  │   (ID, class, zone,        │                  │
│  │    first_seen, last_seen)  │                  │
│  │                            │                  │
│  │   Event Timeline           │                  │
│  │   (appeared, moved,        │                  │
│  │    disappeared + times)    │                  │
│  │                            │                  │
│  │   Zone Tracking            │                  │
│  │   (top-left → bottom-right)│                  │
│  └─────────────┬──────────────┘                  │
│                │                                  │
│                ▼                                  │
│  Llama 4 Scout via OpenRouter                    │
│  + Tool Calling (search_memory, get_timeline)    │
│                │                                  │
│                ▼                                  │
│  ElevenLabs TTS + Deepgram STT                   │
└──────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Video Transport | Vision Agents SDK + Stream Edge (<30ms latency) |
| Object Detection | YOLO26 Nano — 130ms/frame on CPU |
| Object Tracking | ByteTrack — persistent IDs across frames |
| LLM | Llama 4 Scout via OpenRouter |
| TTS | ElevenLabs Flash v2.5 |
| STT | Deepgram with eager turn detection |
| Memory Engine | Custom Python — `temporal_memory.py` |

---

## Project Structure

```
argus/
├── main.py               # Entry point
├── argus_agent.py        # Agent setup — LLM, TTS, STT, tools
├── argus_processor.py    # VideoProcessorPublisher — YOLO + memory pipeline
├── temporal_memory.py    # Temporal Memory Engine (the core innovation)
├── instructions.md       # Agent system prompt
├── diagnostic.py         # Check your setup before running
├── yolo_benchmark.py     # Benchmark YOLO models on your hardware
├── .env.example          # Copy to .env and fill in your keys
└── pyproject.toml        # Dependencies
```

---

## Setup

**Requirements:** Python 3.12+, a webcam, API keys for Stream, OpenRouter, ElevenLabs, Deepgram

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/argus.git
cd argus

# 2. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create environment
uv venv && source .venv/bin/activate   # Linux/Mac
# or: .venv\Scripts\activate           # Windows

# 4. Install dependencies
uv pip install -e .

# 5. Configure API keys
cp .env.example .env
# Edit .env and fill in your keys

# 6. (Optional) Check your setup
uv run diagnostic.py

# 7. Run ARGUS
uv run main.py
```

**Mac users — install this first:**
```bash
brew install portaudio
```

---

## How It Works

### Temporal Memory Engine

Every YOLO detection flows through `TemporalMemory.update_detection()`. Every appearance, disappearance, and movement is logged with timestamps and spatial zones:

```python
# Bounding boxes are converted to human-readable zones
# (342, 256, 89, 112) → "bottom-right"

# Events logged automatically:
# [14:03:12] cup appeared at bottom-right
# [14:05:44] cup moved from bottom-left to bottom-right
# [14:08:01] cup left the scene (was at bottom-right)
```

### LLM Tool Calling

When you ask a question, the LLM calls `search_memory()` — a registered tool that returns the full session history as structured context. The LLM never re-analyzes video. It reads memory.

```python
@llm.register_function(description="Search ARGUS memory for objects or events")
async def search_memory(query: str) -> dict:
    return {"result": argus.memory.build_context(query)}
```

### Zone-Based Spatial Reasoning

Instead of raw pixel coordinates, ARGUS converts detections into 9 human-readable zones (top-left, middle-center, bottom-right, etc.) — making responses natural and readable.

---

## YOLO Benchmark Results

Tested on CPU (4 cores, 15.5 GB RAM):

| Model | Speed | Max FPS |
|-------|-------|---------|
| **YOLO26 Nano** | **130ms** | **7.7** |
| YOLOv8 Nano | 138ms | 7.2 |
| YOLO11 Nano | 139ms | 7.2 |
| YOLO11 Small | 310ms | 3.2 |

Run `yolo_benchmark.py` to benchmark on your own hardware.

---

## Built For

[Vision Possible: Agent Protocol Hackathon](https://www.wemakedevs.org/hackathons/vision) by WeMakeDevs

Powered by [Vision Agents SDK](https://github.com/GetStream/Vision-Agents) by Stream ⭐