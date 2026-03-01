import sys
import os
import time
import platform

print("=" * 60)
print("ARGUS SYSTEM DIAGNOSTIC")
print("=" * 60)

# 1. System Info
print("\n[1] SYSTEM INFO")
print(f"  OS: {platform.system()} {platform.release()}")
print(f"  Python: {sys.version}")
print(f"  CPU cores: {os.cpu_count()}")

# 2. RAM
try:
    import psutil
    ram = psutil.virtual_memory()
    print(f"\n[2] MEMORY")
    print(f"  Total RAM: {ram.total / (1024**3):.1f} GB")
    print(f"  Available: {ram.available / (1024**3):.1f} GB")
    print(f"  Used: {ram.percent}%")
    if ram.available < 2 * (1024**3):
        print("  ⚠️ LOW RAM — may cause lag")
    else:
        print("  ✅ RAM OK")
except:
    print("\n[2] MEMORY: Could not check")

# 3. GPU
print(f"\n[3] GPU CHECK")
device = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  ✅ CUDA GPU: {gpu_name}")
        print(f"  GPU Memory: {gpu_mem:.1f} GB")
        device = "cuda"
    else:
        print("  ❌ No CUDA GPU — using CPU")
except:
    print("  ❌ Could not check GPU")

# 4. YOLO Speed Test
print(f"\n[4] YOLO SPEED TEST (device={device})")
recommended_fps = 3
try:
    from ultralytics import YOLO
    import numpy as np
    model = YOLO("yolo11n.pt")
    model.to(device)
    fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    model.track(fake_frame, persist=True, tracker="bytetrack.yaml", conf=0.4, device=device, verbose=False)
    times = []
    for i in range(5):
        start = time.time()
        model.track(fake_frame, persist=True, tracker="bytetrack.yaml", conf=0.4, device=device, verbose=False)
        times.append(time.time() - start)
    avg_ms = sum(times) / len(times) * 1000
    fps_possible = 1000 / avg_ms
    print(f"  Avg detection: {avg_ms:.0f}ms per frame")
    print(f"  Max FPS: {fps_possible:.1f}")
    if avg_ms > 500:
        recommended_fps = 2
    elif avg_ms > 200:
        recommended_fps = 3
    else:
        recommended_fps = 5
    print(f"  RECOMMENDED FPS: {recommended_fps}")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# 5. API Keys
print(f"\n[5] API KEYS")
from dotenv import load_dotenv
load_dotenv()
keys = ["STREAM_API_KEY", "STREAM_API_SECRET", "OPENROUTER_API_KEY", "ELEVENLABS_API_KEY", "DEEPGRAM_API_KEY", "GOOGLE_API_KEY"]
for k in keys:
    v = os.getenv(k, "")
    status = "✅" if len(v) > 5 else "❌ MISSING"
    print(f"  {status} {k}")

# 6. Memory Engine
print(f"\n[6] TEMPORAL MEMORY")
try:
    from temporal_memory import TemporalMemory
    m = TemporalMemory()
    m.update_detection(1, "cup", (100,200,50,50), 640, 480)
    m.update_detection(2, "person", (300,100,100,200), 640, 480)
    m.update_detection(1, "cup", (400,200,50,50), 640, 480)
    print(f"  Objects: {m.get_stats()['total_objects']}")
    print(f"  Events: {m.get_stats()['total_events']}")
    print("  ✅ WORKS")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# 7. Processor
print(f"\n[7] PROCESSOR")
try:
    import asyncio
    async def t():
        from argus_processor import ArgusProcessor
        p = ArgusProcessor(fps=recommended_fps, yolo_model="yolo11n.pt", device=device)
        print(f"  ✅ Created (fps={recommended_fps}, device={device})")
    asyncio.run(t())
except Exception as e:
    print(f"  ❌ Failed: {e}")

# Summary
print(f"\n{'=' * 60}")
print("APPLY THESE SETTINGS IN argus_agent.py:")
print(f"{'=' * 60}")
print(f"  fps={recommended_fps}")
print(f"  device='{device}'")
print(f"{'=' * 60}")