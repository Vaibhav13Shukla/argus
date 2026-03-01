import time
import numpy as np
from ultralytics import YOLO

fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

models = [
    "yolo11n.pt",      # What we're using now (nano)
    "yolov8n.pt",      # v8 nano
]

# Try to add latest models
try_models = [
    "yolo11s.pt",      # v11 small (more accurate, slower)
    "yolov12n.pt",     # v12 nano (attention-based)
    "yolo26n.pt",      # v26 nano (LATEST)
]

for m in try_models:
    try:
        YOLO(m)  # This triggers download
        models.append(m)
    except Exception as e:
        print(f"  {m}: Not available ({type(e).__name__})")

print("=" * 60)
print("YOLO MODEL BENCHMARK (CPU)")
print("=" * 60)

results = []

for model_name in models:
    try:
        print(f"\nTesting {model_name}...")
        model = YOLO(model_name)
        model.to("cpu")
        
        # Warmup
        model.track(fake_frame, persist=True, tracker="bytetrack.yaml",
                   conf=0.4, device="cpu", verbose=False)
        
        # Benchmark 5 frames
        times = []
        for i in range(5):
            start = time.time()
            r = model.track(fake_frame, persist=True, tracker="bytetrack.yaml",
                          conf=0.4, device="cpu", verbose=False)
            times.append(time.time() - start)
        
        avg_ms = sum(times) / len(times) * 1000
        max_fps = 1000 / avg_ms
        
        # Count detectable classes
        num_classes = len(model.names)
        
        print(f"  ✅ {model_name}")
        print(f"     Speed: {avg_ms:.0f}ms/frame")
        print(f"     Max FPS: {max_fps:.1f}")
        print(f"     Classes: {num_classes}")
        
        results.append((model_name, avg_ms, max_fps, num_classes))
        
    except Exception as e:
        print(f"  ❌ {model_name}: {e}")

print(f"\n{'=' * 60}")
print("RESULTS (sorted by speed)")
print(f"{'=' * 60}")
print(f"{'Model':<20} {'Speed':<12} {'Max FPS':<10} {'Classes':<10}")
print("-" * 52)

for name, ms, fps, cls in sorted(results, key=lambda x: x[1]):
    marker = " ← FASTEST" if ms == min(r[1] for r in results) else ""
    print(f"{name:<20} {ms:.0f}ms{'':<7} {fps:.1f}{'':<6} {cls}{marker}")

best = min(results, key=lambda x: x[1])
print(f"\n🏆 BEST FOR YOUR CPU: {best[0]} ({best[1]:.0f}ms, {best[2]:.1f} FPS)")
print(f"\nUse this in argus_agent.py:")
print(f'  yolo_model="{best[0]}"')