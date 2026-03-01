import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

logger = logging.getLogger("argus.memory")
TIMESTAMP_FORMAT = "%H:%M:%S"


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    last_bbox: tuple = (0, 0, 0, 0)
    last_zone: str = "unknown"
    is_active: bool = True
    detection_count: int = 1

    def to_summary(self, now: float) -> str:
        status = "VISIBLE" if self.is_active else "GONE"
        ago = now - self.last_seen
        t = time.strftime(TIMESTAMP_FORMAT, time.localtime(self.last_seen))
        if ago < 2:
            when = "right now"
        elif ago < 60:
            when = f"{ago:.0f}s ago"
        elif ago < 3600:
            when = f"{ago/60:.0f}min ago"
        else:
            when = f"at {t}"
        return f"[{status}] {self.class_name} (ID:{self.track_id}) at {self.last_zone}, seen {when}, detected {self.detection_count}x"


@dataclass
class ActionEvent:
    timestamp: float
    event_type: str
    description: str
    track_id: int = -1
    class_name: str = ""

    @property
    def time_str(self) -> str:
        return time.strftime(TIMESTAMP_FORMAT, time.localtime(self.timestamp))

    def to_natural(self) -> str:
        return f"[{self.time_str}] {self.description}"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.time_str,
            "event_type": self.event_type,
            "description": self.description,
            "class_name": self.class_name,
        }


class TemporalMemory:
    def __init__(self, disappear_threshold: float = 5.0):
        self.objects: Dict[int, TrackedObject] = {}
        self.events: deque[ActionEvent] = deque(maxlen=10000)
        self.scene_descriptions: deque[tuple[float, str]] = deque(maxlen=200)
        self.disappear_threshold = disappear_threshold
        self.session_start = time.time()
        self.frame_count = 0

    def update_detection(self, track_id: int, class_name: str, bbox: tuple, frame_width: int, frame_height: int):
        now = time.time()
        zone = self._bbox_to_zone(bbox, frame_width, frame_height)
        if track_id not in self.objects:
            self.objects[track_id] = TrackedObject(
                track_id=track_id, class_name=class_name,
                first_seen=now, last_seen=now, last_bbox=bbox,
                last_zone=zone, is_active=True, detection_count=1,
            )
            self._log("appeared", f"{class_name} appeared at {zone}", track_id, class_name)
            logger.info(f"NEW: {class_name} ID:{track_id} at {zone}")
        else:
            obj = self.objects[track_id]
            old_zone = obj.last_zone
            if not obj.is_active:
                obj.is_active = True
                obj.detection_count += 1
                self._log("reappeared", f"{class_name} (ID:{track_id}) reappeared at {zone}", track_id, class_name)
            obj.last_seen = now
            obj.last_bbox = bbox
            obj.last_zone = zone
            obj.detection_count += 1
            if old_zone != zone:
                self._log("moved", f"{class_name} (ID:{track_id}) moved from {old_zone} to {zone}", track_id, class_name)

    def check_disappearances(self):
        now = time.time()
        for obj in self.objects.values():
            if obj.is_active and (now - obj.last_seen) > self.disappear_threshold:
                obj.is_active = False
                self._log("disappeared", f"{obj.class_name} (ID:{obj.track_id}) left the scene (was at {obj.last_zone})", obj.track_id, obj.class_name)
                logger.info(f"GONE: {obj.class_name} ID:{obj.track_id}")

    def add_scene_description(self, description: str):
        self.scene_descriptions.append((time.time(), description))

    def build_context(self, query: str = "") -> str:
        now = time.time()
        parts = []
        elapsed = now - self.session_start
        active_count = sum(1 for o in self.objects.values() if o.is_active)
        parts.append(f"[ARGUS MEMORY] Time: {time.strftime(TIMESTAMP_FORMAT, time.localtime(now))} | Session: {elapsed/60:.1f}min | Tracked: {len(self.objects)} | Visible: {active_count} | Events: {len(self.events)}")
        active = [o for o in self.objects.values() if o.is_active]
        if active:
            parts.append("\n[CURRENTLY VISIBLE]")
            for obj in active:
                parts.append(f"  {obj.to_summary(now)}")
        gone = [o for o in self.objects.values() if not o.is_active and (now - o.last_seen) < 300]
        if gone:
            parts.append("\n[RECENTLY LEFT]")
            for obj in sorted(gone, key=lambda o: o.last_seen, reverse=True)[:10]:
                parts.append(f"  {obj.to_summary(now)}")
        recent = [e for e in self.events if now - e.timestamp < 300][-30:]
        if recent:
            parts.append("\n[RECENT EVENTS]")
            for event in recent:
                parts.append(f"  {event.to_natural()}")
        if self.scene_descriptions:
            ts, desc = self.scene_descriptions[-1]
            t = time.strftime(TIMESTAMP_FORMAT, time.localtime(ts))
            parts.append(f"\n[SCENE at {t}] {desc}")
        return "\n".join(parts)

    def find_object(self, object_type: str) -> str:
        now = time.time()
        matches = [o for o in self.objects.values() if object_type.lower() in o.class_name.lower()]
        if not matches:
            return f"No '{object_type}' seen during this session."
        return "\n".join(obj.to_summary(now) for obj in sorted(matches, key=lambda o: o.last_seen, reverse=True))

    def get_timeline(self, minutes: int = 10) -> List[dict]:
        cutoff = time.time() - (minutes * 60)
        return [e.to_dict() for e in self.events if e.timestamp >= cutoff]

    def get_stats(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "session_minutes": round((now - self.session_start) / 60, 1),
            "total_objects": len(self.objects),
            "active_objects": sum(1 for o in self.objects.values() if o.is_active),
            "total_events": len(self.events),
            "frames_processed": self.frame_count,
        }

    def _log(self, event_type, description, track_id, class_name):
        self.events.append(ActionEvent(timestamp=time.time(), event_type=event_type, description=description, track_id=track_id, class_name=class_name))

    def _bbox_to_zone(self, bbox, fw, fh) -> str:
        x, y, w, h = bbox
        cx = (x + w / 2) / fw if fw > 0 else 0.5
        cy = (y + h / 2) / fh if fh > 0 else 0.5
        col = "left" if cx < 0.33 else ("center" if cx < 0.66 else "right")
        row = "top" if cy < 0.33 else ("middle" if cy < 0.66 else "bottom")
        return f"{row}-{col}"
