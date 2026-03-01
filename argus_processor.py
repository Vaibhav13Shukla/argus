import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import av
import cv2
import numpy as np

from vision_agents.core.events.base import PluginBaseEvent
from vision_agents.core.events.manager import EventManager
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.core.warmup import Warmable

from temporal_memory import TemporalMemory

logger = logging.getLogger("argus.processor")


@dataclass
class ObjectAppearedEvent(PluginBaseEvent):
    type: str = field(default="argus.object_appeared", init=False)
    track_id: int = 0
    class_name: str = ""
    zone: str = ""

@dataclass
class ObjectDisappearedEvent(PluginBaseEvent):
    type: str = field(default="argus.object_disappeared", init=False)
    track_id: int = 0
    class_name: str = ""
    last_zone: str = ""

@dataclass
class ObjectMovedEvent(PluginBaseEvent):
    type: str = field(default="argus.object_moved", init=False)
    track_id: int = 0
    class_name: str = ""
    from_zone: str = ""
    to_zone: str = ""


class ArgusProcessor(VideoProcessorPublisher, Warmable[Optional[Any]]):
    name = "argus"

    def __init__(self, fps: int = 5, yolo_model: str = "yolo11n.pt", device: str = "cpu", conf_threshold: float = 0.4, disappear_threshold: float = 5.0, max_workers: int = 4):
        self.fps = fps
        self.yolo_model_path = yolo_model
        self.device = device
        self.conf_threshold = conf_threshold
        self.max_workers = max_workers
        self.yolo_model = None
        self.memory = TemporalMemory(disappear_threshold=disappear_threshold)
        self._video_track = QueuedVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="argus")
        self._shutdown = False
        self.events = EventManager()
        self.events.register(ObjectAppearedEvent)
        self.events.register(ObjectDisappearedEvent)
        self.events.register(ObjectMovedEvent)
        self._prev_active_ids: set = set()
        logger.info(f"ARGUS Processor initialized (fps={fps}, model={yolo_model})")

    async def on_warmup(self) -> Optional[Any]:
        try:
            from ultralytics import YOLO
            loop = asyncio.get_event_loop()
            def load_model():
                model = YOLO(self.yolo_model_path)
                model.to(self.device)
                logger.info(f"YOLO model loaded: {self.yolo_model_path}")
                return model
            return await loop.run_in_executor(self.executor, load_model)
        except Exception as e:
            logger.error(f"YOLO load failed: {e}")
            return None

    def on_warmed_up(self, resource: Optional[Any]) -> None:
        self.yolo_model = resource

    async def process_video(self, track, participant_id: Optional[str], shared_forwarder: Optional[VideoForwarder] = None) -> None:
        if shared_forwarder is not None:
            self._video_forwarder = shared_forwarder
            self._video_forwarder.add_frame_handler(self._process_frame, fps=float(self.fps), name="argus")
        else:
            self._video_forwarder = VideoForwarder(track, max_buffer=30, fps=self.fps, name="argus_forwarder")
            self._video_forwarder.add_frame_handler(self._process_frame)
        logger.info("ARGUS video processing started")

    async def stop_processing(self) -> None:
        if self._video_forwarder:
            await self._video_forwarder.stop()

    def publish_video_track(self):
        return self._video_track

    async def _process_frame(self, frame: av.VideoFrame):
        try:
            frame_rgb = frame.to_ndarray(format="rgb24")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            height, width = frame_bgr.shape[:2]
            self.memory.frame_count += 1

            if self.yolo_model is not None:
                loop = asyncio.get_event_loop()
                detections = await loop.run_in_executor(self.executor, self._run_yolo_sync, frame_bgr)
                current_ids = set()
                for det in detections:
                    track_id = det["track_id"]
                    current_ids.add(track_id)
                    old_zone = None
                    if track_id in self.memory.objects:
                        old_zone = self.memory.objects[track_id].last_zone
                    self.memory.update_detection(track_id=track_id, class_name=det["class_name"], bbox=det["bbox"], frame_width=width, frame_height=height)
                    new_zone = self.memory.objects[track_id].last_zone
                    if track_id not in self._prev_active_ids:
                        self.events.send(ObjectAppearedEvent(plugin_name="argus", track_id=track_id, class_name=det["class_name"], zone=new_zone))
                    if old_zone and old_zone != new_zone:
                        self.events.send(ObjectMovedEvent(plugin_name="argus", track_id=track_id, class_name=det["class_name"], from_zone=old_zone, to_zone=new_zone))
                self._prev_active_ids = current_ids
                frame_bgr = self._draw_detections(frame_bgr, detections)

            self.memory.check_disappearances()
            frame_bgr = self._draw_status(frame_bgr)
            frame_rgb_out = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            processed_frame = av.VideoFrame.from_ndarray(frame_rgb_out, format="rgb24")
            await self._video_track.add_frame(processed_frame)
        except Exception as e:
            logger.exception(f"Frame processing failed: {e}")
            await self._video_track.add_frame(frame)

    def _run_yolo_sync(self, frame_bgr: np.ndarray) -> list:
        detections = []
        try:
            results = self.yolo_model.track(frame_bgr, persist=True, tracker="bytetrack.yaml", conf=self.conf_threshold, device=self.device, verbose=False)
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    track_id = int(boxes.id[i])
                    class_id = int(boxes.cls[i])
                    class_name = self.yolo_model.names[class_id]
                    confidence = float(boxes.conf[i])
                    x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    detections.append({"track_id": track_id, "class_name": class_name, "confidence": confidence, "bbox": (x, y, w, h), "xyxy": (x1, y1, x2, y2)})
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
        return detections

    def _draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        COLORS = [(0,255,170),(255,100,100),(100,100,255),(255,255,100),(255,100,255),(100,255,255),(200,150,50),(150,255,100)]
        for det in detections:
            tid = det["track_id"]
            x1, y1, x2, y2 = det["xyxy"]
            color = COLORS[tid % len(COLORS)]
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"ID:{tid} {det['class_name']} {det['confidence']:.0%}"
            ls = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(frame, (x1, y1-ls[1]-8), (x1+ls[0]+4, y1), color, -1)
            cv2.putText(frame, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
            obj = self.memory.objects.get(tid)
            if obj:
                cv2.putText(frame, f"@ {obj.last_zone}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        return frame

    def _draw_status(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        stats = self.memory.get_stats()
        cv2.rectangle(frame, (0,0), (w,28), (0,0,0), -1)
        status = f"ARGUS | V:{stats['active_objects']} T:{stats['total_objects']} E:{stats['total_events']}"
        cv2.putText(frame, status, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,170), 1)
        return frame

    def state(self) -> Dict[str, Any]:
        stats = self.memory.get_stats()
        return {"argus_memory_context": self.memory.build_context(), **stats}

    async def close(self):
        self._shutdown = True
        if self._video_forwarder:
            await self._video_forwarder.stop()
        self.executor.shutdown(wait=False)
        logger.info("ARGUS processor closed")
