"""
detector.py — Face Detection & Face Tracking
==============================================
AI-Powered Face Detection System

Contains:
  - DetectionResult  : dataclass holding all detections for one frame
  - CentroidTracker  : lightweight tracker that assigns stable IDs to faces
  - FaceDetector     : loads Haar Cascades, detects faces/eyes/smiles,
                       computes pseudo-confidence scores
"""

import cv2
import os
import shutil
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from collections import OrderedDict

import config

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────────
#  DATA CLASS
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """All detection data produced from a single video frame."""
    faces:    List[Tuple[int, int, int, int]] = field(default_factory=list)
    eyes:     List[Tuple[int, int, int, int]] = field(default_factory=list)
    smiles:   List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Mapping: face_id → (x, y, w, h) for tracked faces
    face_ids: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────────────
#  CENTROID TRACKER
# ───────────────────────────────────────────────────────────────────────────────

class CentroidTracker:
    """
    Assigns a persistent integer ID to each face across frames by matching
    face centroids using nearest-neighbour assignment.

    Algorithm (per frame):
      1. Compute centroids of newly detected bounding boxes.
      2. If no existing tracks: register all as new.
      3. Otherwise: build a distance matrix (existing × new centroids),
         greedily match pairs within MAX_DISTANCE threshold.
      4. Unmatched existing tracks: increment disappeared counter;
         deregister when counter exceeds MAX_DISAPPEARED.
      5. Unmatched new detections: register as fresh tracks.
    """

    def __init__(self) -> None:
        self.next_id    : int                                  = 0
        self.objects    : OrderedDict[int, Tuple[int, int]]    = OrderedDict()
        self.rects      : OrderedDict[int, Tuple[int,int,int,int]] = OrderedDict()
        self.disappeared: OrderedDict[int, int]                = OrderedDict()

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _centroid(rect: Tuple[int,int,int,int]) -> Tuple[int,int]:
        x, y, w, h = rect
        return (int(x + w / 2), int(y + h / 2))

    def _register(self, rect: Tuple[int,int,int,int]) -> None:
        self.objects[self.next_id]      = self._centroid(rect)
        self.rects[self.next_id]        = rect
        self.disappeared[self.next_id]  = 0
        self.next_id += 1

    def _deregister(self, oid: int) -> None:
        del self.objects[oid]
        del self.rects[oid]
        del self.disappeared[oid]

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, rects: list) -> Dict[int, Tuple[int,int,int,int]]:
        """
        Feed new detections; get back a dict of {face_id: bounding_rect}.
        Pass an empty list when no faces are detected.
        """
        # ── No detections: age all existing tracks ─────────────────────────
        if not rects:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > config.TRACKER_MAX_DISAPPEARED:
                    self._deregister(oid)
            return dict(self.rects)

        new_cents = [self._centroid(r) for r in rects]

        # ── No existing tracks: register everything fresh ──────────────────
        if not self.objects:
            for r in rects:
                self._register(r)
            return dict(self.rects)

        # ── Match new centroids to existing tracks ─────────────────────────
        oids      = list(self.objects.keys())
        old_cents = list(self.objects.values())

        # Distance matrix: rows = existing, cols = new detections
        D = np.zeros((len(old_cents), len(new_cents)), dtype=float)
        for i, oc in enumerate(old_cents):
            for j, nc in enumerate(new_cents):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(nc))

        # Greedy matching: sort rows by minimum distance, assign best column
        row_order = D.min(axis=1).argsort()
        col_order = D.argmin(axis=1)[row_order]

        used_rows: set = set()
        used_cols: set = set()

        for r, c in zip(row_order, col_order):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > config.TRACKER_MAX_DISTANCE:
                continue
            oid = oids[r]
            self.objects[oid]      = new_cents[c]
            self.rects[oid]        = rects[c]
            self.disappeared[oid]  = 0
            used_rows.add(r)
            used_cols.add(c)

        # Unmatched existing tracks → age them
        for r in set(range(len(old_cents))) - used_rows:
            oid = oids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > config.TRACKER_MAX_DISAPPEARED:
                self._deregister(oid)

        # Unmatched new detections → register as new faces
        for c in set(range(len(new_cents))) - used_cols:
            self._register(rects[c])

        return dict(self.rects)


# ───────────────────────────────────────────────────────────────────────────────
#  FACE DETECTOR
# ───────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    Real-time face detector built on OpenCV Haar Cascade classifiers.

    Features:
      • Multi-face detection
      • Eye detection within each face region-of-interest
      • Smile detection within the lower face ROI
      • Persistent face-ID tracking via CentroidTracker
      • Pseudo-confidence score per detected face
    """

    def __init__(self) -> None:
        self.face_cascade  : cv2.CascadeClassifier | None = None
        self.eye_cascade   : cv2.CascadeClassifier | None = None
        self.smile_cascade : cv2.CascadeClassifier | None = None
        self.tracker = CentroidTracker()
        self._load_cascades()

    # ── cascade loading ───────────────────────────────────────────────────────

    def _resolve_cascade(self, filename: str) -> str:
        """
        Return a valid path to the cascade XML.
        Prefers a local copy; falls back to OpenCV's bundled data directory
        and copies the file locally for portability.
        """
        if os.path.exists(filename):
            return filename

        bundled = os.path.join(cv2.data.haarcascades, filename)
        if os.path.exists(bundled):
            shutil.copy2(bundled, filename)
            log.info(f"Cascade copied from OpenCV bundle: {filename}")
            return filename

        raise FileNotFoundError(
            f"Cascade file not found: '{filename}'. "
            "Ensure opencv-python is installed correctly."
        )

    def _load_cascades(self) -> None:
        # ── Face (required) ───────────────────────────────────────
        path = self._resolve_cascade(config.CASCADE_FACE)
        self.face_cascade = cv2.CascadeClassifier(path)
        if self.face_cascade.empty():
            raise RuntimeError(
                "Face cascade loaded but is empty — "
                "the XML file may be corrupt. Re-install opencv-python."
            )
        log.info("✅ Face cascade loaded successfully.")

        # ── Eye & smile (optional — detection disabled gracefully if absent) ─
        for filename, attr in [
            (config.CASCADE_EYE,   "eye_cascade"),
            (config.CASCADE_SMILE, "smile_cascade"),
        ]:
            try:
                cascade = cv2.CascadeClassifier(self._resolve_cascade(filename))
                setattr(self, attr, cascade)
                log.info(f"✅ {filename} loaded.")
            except Exception as exc:
                log.warning(f"Optional cascade skipped ({filename}): {exc}")

    # ── detection ─────────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        detect_eyes:   bool = True,
        detect_smiles: bool = True,
    ) -> DetectionResult:
        """
        Detect faces (and optionally eyes / smiles) in a BGR frame.

        Steps:
          1. Convert to grayscale + histogram equalisation for lighting robustness.
          2. Run face cascade on the full image.
          3. For each face, run eye cascade on the upper half of the face ROI.
          4. For each face, run smile cascade on the lower half of the face ROI.
          5. Update the CentroidTracker and populate DetectionResult.

        Returns:
            DetectionResult populated with all detected rectangles and face IDs.
        """
        result = DetectionResult()

        # ── Pre-process ───────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # Improve detection in varied lighting

        # ── Face detection ────────────────────────────────────────
        raw_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor  = config.FACE_SCALE_FACTOR,
            minNeighbors = config.FACE_MIN_NEIGHBORS,
            minSize      = config.FACE_MIN_SIZE,
            flags        = cv2.CASCADE_SCALE_IMAGE,
        )

        if len(raw_faces) == 0:
            # No faces — update tracker with empty list so IDs age out
            result.face_ids = self.tracker.update([])
            return result

        face_list = [tuple(int(v) for v in r) for r in raw_faces]
        result.faces    = face_list
        result.face_ids = self.tracker.update(face_list)

        # ── Per-face: eye & smile detection ───────────────────────
        for (x, y, w, h) in face_list:
            face_gray = gray[y:y + h, x:x + w]

            # Eyes — search only upper half to avoid false positives
            if (detect_eyes
                    and self.eye_cascade is not None
                    and not self.eye_cascade.empty()):
                eyes = self.eye_cascade.detectMultiScale(
                    face_gray[: h // 2],
                    scaleFactor  = config.EYE_SCALE_FACTOR,
                    minNeighbors = config.EYE_MIN_NEIGHBORS,
                    minSize      = config.EYE_MIN_SIZE,
                )
                for (ex, ey, ew, eh) in eyes:
                    result.eyes.append((x + ex, y + ey, ew, eh))

            # Smile — search only lower half of face
            if (detect_smiles
                    and self.smile_cascade is not None
                    and not self.smile_cascade.empty()):
                smiles = self.smile_cascade.detectMultiScale(
                    face_gray[h // 2 :],
                    scaleFactor  = config.SMILE_SCALE_FACTOR,
                    minNeighbors = config.SMILE_MIN_NEIGHBORS,
                    minSize      = config.SMILE_MIN_SIZE,
                )
                for (sx, sy, sw, sh) in smiles:
                    result.smiles.append((x + sx, y + h // 2 + sy, sw, sh))

        return result

    def compute_confidence(
        self, rect: Tuple[int,int,int,int], frame_shape: Tuple
    ) -> float:
        """
        Return a pseudo-confidence score 0–100 for a detected face.

        Score is derived from:
          • Size  (60 %): larger face relative to frame → higher score.
          • Position (40 %): closer to frame centre → higher score.

        Not a true ML probability, but useful as a human-readable quality hint.
        """
        x, y, w, h   = rect
        fh, fw        = frame_shape[:2]
        size_score    = min(1.0, (w * h) / max(1, fw * fh * 0.30)) * 60
        cx, cy        = x + w / 2, y + h / 2
        dist          = ((cx - fw / 2) ** 2 + (cy - fh / 2) ** 2) ** 0.5
        max_dist      = ((fw / 2) ** 2 + (fh / 2) ** 2) ** 0.5
        center_score  = (1.0 - dist / max_dist) * 40
        return round(min(100.0, size_score + center_score), 1)