"""Circular Ring Buffer Manager for Decoded Video Frames.

Implements bounded VRAM / pinned memory ring buffers (B_ring <= 16) per stream channel,
ensuring O(1) frame ingestion, strictly constant memory consumption, and lock-free/mutex-safe
access for downstream inference consumers (RT-DETR and ByteTrack).
"""

import threading
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple, List


class VRAMRingBuffer:
    """Bounded circular ring buffer storing decoded video frames and metadata."""

    def __init__(self, capacity: int = 16, channel_id: str = "CAM-DEFAULT"):
        self.capacity = min(max(capacity, 4), 64)
        self.channel_id = channel_id
        self._lock = threading.Lock()

        # Preallocated ring slots
        self._buffer: List[Optional[np.ndarray]] = [None] * self.capacity
        self._timestamps: List[float] = [0.0] * self.capacity
        self._frame_indices: List[int] = [0] * self.capacity

        self._head = 0  # Write pointer
        self._tail = 0  # Read pointer
        self._count = 0
        self._dropped_overflow_count = 0

    def push(self, frame: np.ndarray, timestamp: Optional[float] = None, frame_idx: int = 0):
        """Insert a newly decoded frame into the ring buffer.

        If the buffer is full, automatically advances the tail pointer,
        dropping the oldest frame to preserve real-time low latency.
        """
        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            if self._count == self.capacity:
                # Buffer overflow: drop oldest frame
                self._tail = (self._tail + 1) % self.capacity
                self._dropped_overflow_count += 1
            else:
                self._count += 1

            self._buffer[self._head] = frame
            self._timestamps[self._head] = ts
            self._frame_indices[self._head] = frame_idx
            self._head = (self._head + 1) % self.capacity

    def pop(self) -> Tuple[Optional[np.ndarray], float, int]:
        """Retrieve and remove the oldest frame from the buffer."""
        with self._lock:
            if self._count == 0:
                return None, 0.0, -1

            frame = self._buffer[self._tail]
            ts = self._timestamps[self._tail]
            f_idx = self._frame_indices[self._tail]

            self._buffer[self._tail] = None  # Free slot reference
            self._tail = (self._tail + 1) % self.capacity
            self._count -= 1

            return frame, ts, f_idx

    def peek_latest(self) -> Tuple[Optional[np.ndarray], float, int]:
        """Inspect the most recently pushed frame without popping it."""
        with self._lock:
            if self._count == 0:
                return None, 0.0, -1

            latest_idx = (self._head - 1 + self.capacity) % self.capacity
            return self._buffer[latest_idx], self._timestamps[latest_idx], self._frame_indices[latest_idx]

    def clear(self):
        """Flush all buffered frames."""
        with self._lock:
            self._buffer = [None] * self.capacity
            self._head = 0
            self._tail = 0
            self._count = 0

    @property
    def size(self) -> int:
        with self._lock:
            return self._count

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "channel_id": self.channel_id,
                "capacity": self.capacity,
                "current_fill": self._count,
                "fill_percentage": round((self._count / self.capacity) * 100.0, 1),
                "dropped_overflows": self._dropped_overflow_count
            }


class FrameBufferManager:
    """Central registry of ring buffers for all active municipal camera streams."""

    def __init__(self, default_capacity: int = 16):
        self.default_capacity = default_capacity
        self._buffers: Dict[str, VRAMRingBuffer] = {}
        self._lock = threading.Lock()

    def get_or_create(self, camera_id: str, capacity: Optional[int] = None) -> VRAMRingBuffer:
        with self._lock:
            if camera_id not in self._buffers:
                cap = capacity or self.default_capacity
                self._buffers[camera_id] = VRAMRingBuffer(capacity=cap, channel_id=camera_id)
            return self._buffers[camera_id]

    def remove(self, camera_id: str):
        with self._lock:
            if camera_id in self._buffers:
                self._buffers[camera_id].clear()
                del self._buffers[camera_id]

    def get_all_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {cid: buf.get_stats() for cid, buf in self._buffers.items()}


buffer_manager = FrameBufferManager()
