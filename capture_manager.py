import cv2
import os
import time
import logging
from typing import List

import config
from detector import DetectionResult

log = logging.getLogger(__name__)


class CaptureManager:
    """
    Manages saving images to disk.

    Usage:
        manager = CaptureManager()
        manager.save_screenshot(frame)             # manual screenshot
        manager.save_face_snapshot(frame, rect, 1) # crop + save face #1
        saved = manager.update(frame, detection)   # auto-capture tick
    """

    def __init__(self) -> None:
        self._screenshots_dir = os.path.join(
            config.CAPTURES_DIR, config.SCREENSHOT_SUBDIR
        )
        self._snapshots_dir = os.path.join(
            config.CAPTURES_DIR, config.SNAPSHOT_SUBDIR
        )
        self._auto_mode      : bool  = False
        self._last_capture_t : float = 0.0
        self._total_saved    : int   = 0

        self._ensure_dirs()

    # Setup 

    def _ensure_dirs(self) -> None:
        for d in (self._screenshots_dir, self._snapshots_dir):
            os.makedirs(d, exist_ok=True)
        log.info(
            f"Capture dirs ready: "
            f"'{self._screenshots_dir}', '{self._snapshots_dir}'"
        )

    #  Properties 

    @property
    def auto_mode(self) -> bool:
        return self._auto_mode

    @property
    def total_saved(self) -> int:
        return self._total_saved

    def toggle_auto_capture(self) -> bool:
        """Toggle auto-capture on/off.  Returns the new state."""
        self._auto_mode = not self._auto_mode
        state = "ENABLED" if self._auto_mode else "DISABLED"
        log.info(f"Auto-capture {state}.")
        return self._auto_mode

    #  Internal helpers 

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def _write(self, path: str, image) -> bool:
        """Write image to disk with error handling. Returns True on success."""
        try:
            cv2.imwrite(path, image)
            self._total_saved += 1
            log.info(f"Saved [{self._total_saved}]: {path}")
            return True
        except Exception as exc:
            log.error(f"Failed to save {path}: {exc}")
            return False

    #  Public save methods 

    def save_screenshot(self, frame) -> str:
        """
        Save a full-resolution screenshot.

        Returns:
            Absolute path of the saved file, or empty string on failure.
        """
        ts   = self._timestamp()
        path = os.path.join(
            self._screenshots_dir,
            f"{config.SCREENSHOT_PREFIX}_{ts}.jpg",
        )
        ok = self._write(path, frame)
        return os.path.abspath(path) if ok else ""

    def save_face_snapshot(
        self, frame, rect: tuple, face_id: int
    ) -> str:
        """
        Crop a single face (with padding) and save it as a snapshot.

        Args:
            frame   : Full BGR frame.
            rect    : (x, y, w, h) of the detected face.
            face_id : Tracker-assigned integer ID.

        Returns:
            Absolute path of the saved file, or empty string on failure.
        """
        x, y, w, h   = rect
        fh, fw        = frame.shape[:2]
        pad           = 18                          # padding around face crop
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fw, x + w + pad)
        y2 = min(fh, y + h + pad)

        crop = frame[y1:y2, x1:x2]
        ts   = self._timestamp()
        path = os.path.join(
            self._snapshots_dir,
            f"{config.SNAPSHOT_PREFIX}_id{face_id}_{ts}.jpg",
        )
        ok = self._write(path, crop)
        return os.path.abspath(path) if ok else ""

    #  Per-frame update (called inside the main loop) 

    def update(self, frame, detection: DetectionResult) -> List[str]:
        """
        Auto-capture tick — call once per frame.

        When auto-capture is active and enough time has elapsed, saves a
        snapshot for every tracked face in the current detection.

        Returns:
            List of saved file paths (empty if nothing was saved this frame).
        """
        saved: List[str] = []

        if not self._auto_mode or not detection.faces:
            return saved

        now = time.time()
        if now - self._last_capture_t < config.AUTO_CAPTURE_INTERVAL:
            return saved

        self._last_capture_t = now
        for fid, rect in detection.face_ids.items():
            path = self.save_face_snapshot(frame, rect, fid)
            if path:
                saved.append(path)

        return saved
