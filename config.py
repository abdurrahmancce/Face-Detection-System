"""
config.py — Central Configuration
===================================
AI-Powered Face Detection System
All tunable constants live here — modify freely to customise behaviour.
"""

import cv2

# ═══════════════════════════════════════════════════════════════════════════════
#  WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
WINDOW_TITLE   = "AI Face Detection System"
DEFAULT_WIDTH  = 1280
DEFAULT_HEIGHT = 720

# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA
# ═══════════════════════════════════════════════════════════════════════════════
CAMERA_INDEX  = 0       # Change to 1 or 2 if default webcam is not found
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30

# ═══════════════════════════════════════════════════════════════════════════════
#  HAAR CASCADE FILENAMES
# ═══════════════════════════════════════════════════════════════════════════════
CASCADE_FACE  = "haarcascade_frontalface_default.xml"
CASCADE_EYE   = "haarcascade_eye.xml"
CASCADE_SMILE = "haarcascade_smile.xml"

# ═══════════════════════════════════════════════════════════════════════════════
#  FACE DETECTION PARAMETERS
#  Adjust scaleFactor / minNeighbors for accuracy vs speed trade-off:
#    ↓ scaleFactor  → more scales checked → slower but more accurate
#    ↑ minNeighbors → fewer false positives, may miss distant faces
# ═══════════════════════════════════════════════════════════════════════════════
FACE_SCALE_FACTOR  = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE      = (30, 30)

EYE_SCALE_FACTOR   = 1.1
EYE_MIN_NEIGHBORS  = 10
EYE_MIN_SIZE       = (15, 15)

SMILE_SCALE_FACTOR   = 1.8
SMILE_MIN_NEIGHBORS  = 20
SMILE_MIN_SIZE       = (25, 25)

# ═══════════════════════════════════════════════════════════════════════════════
#  COLORS  (OpenCV uses BGR channel order, not RGB)
# ═══════════════════════════════════════════════════════════════════════════════
CLR_FACE_BOX   = (  0, 220,   0)   # Bright green  — face bounding box
CLR_EYE_BOX    = (200, 170,   0)   # Cyan-blue     — eye rectangles
CLR_SMILE_BOX  = (  0, 140, 255)   # Orange        — smile rectangles
CLR_ID_TAG     = (  0, 220, 220)   # Yellow        — face ID label
CLR_CAPTURE    = (  0, 255, 180)   # Teal-green    — capture notification
CLR_ACCENT     = (  0, 200, 255)   # Orange-amber  — title / accent
CLR_WHITE      = (255, 255, 255)
CLR_DARK_BG    = ( 15,  15,  15)   # Near-black overlay

CLR_FPS_GOOD   = (  0, 220,   0)   # ≥ 20 FPS
CLR_FPS_OK     = (  0, 220, 220)   # 10–20 FPS
CLR_FPS_LOW    = (  0,   0, 220)   # < 10 FPS

# ═══════════════════════════════════════════════════════════════════════════════
#  FONTS
# ═══════════════════════════════════════════════════════════════════════════════
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX
FONT_SM    = 0.42
FONT_MD    = 0.58
FONT_LG    = 0.78
FONT_XL    = 1.05
BOX_TH     = 2   # Bounding-box line thickness

# ═══════════════════════════════════════════════════════════════════════════════
#  FACE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
TRACKER_MAX_DISAPPEARED = 15   # Frames a face can vanish before its ID is dropped
TRACKER_MAX_DISTANCE    = 90   # Max pixel distance to re-match a face centroid

# ═══════════════════════════════════════════════════════════════════════════════
#  CAPTURE / SAVE
# ═══════════════════════════════════════════════════════════════════════════════
CAPTURES_DIR           = "captures"
SCREENSHOT_SUBDIR      = "screenshots"
SNAPSHOT_SUBDIR        = "snapshots"
SCREENSHOT_PREFIX      = "screenshot"
SNAPSHOT_PREFIX        = "face_snapshot"
AUTO_CAPTURE_INTERVAL  = 5.0    # Seconds between automatic face snapshots
NOTIFICATION_DURATION  = 70     # Frames the save-notification stays on screen