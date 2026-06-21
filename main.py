import cv2
import sys
import time
import logging

import config
from detector        import FaceDetector, DetectionResult
from renderer        import UIRenderer
from capture_manager import CaptureManager


#  LOGGING SETUP

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("face_detection.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("FaceDetectionApp")


#  APPLICATION

class FaceDetectionApp:
    """
    Main application class.  Orchestrates:
      - Webcam capture
      - Face / eye / smile detection  (FaceDetector)
      - UI overlay rendering          (UIRenderer)
      - Image capture & auto-save     (CaptureManager)
    """

    def __init__(self) -> None:
        # Core components
        self.cap      : cv2.VideoCapture | None = None
        self.detector  = FaceDetector()
        self.renderer  = UIRenderer()
        self.capture   = CaptureManager()

        #  Application state 
        self.running        : bool  = False
        self.detect_eyes    : bool  = True
        self.detect_smiles  : bool  = True
        self.fullscreen     : bool  = False

        #  FPS tracking 
        self._fps           : float = 0.0
        self._fps_counter   : int   = 0
        self._fps_start     : float = time.time()

        #  Latest detection (needed by key handler) 
        self._last_detection: DetectionResult | None = None
        self._current_frame = None          # most recent raw frame

    #  Initialisation  

    def _init_camera(self) -> None:
        """Open the webcam and request the configured resolution."""
        log.info(f"Opening camera (index {config.CAMERA_INDEX}) …")
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {config.CAMERA_INDEX}.\n"
                "  • Check that a webcam is connected.\n"
                "  • Make sure no other application is using the camera.\n"
                "  • Try changing CAMERA_INDEX in config.py (e.g. 1 or 2)."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Camera opened — resolution: {actual_w} × {actual_h}")

    def _init_window(self) -> None:
        """Create and size the display window."""
        cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_TITLE, config.DEFAULT_WIDTH, config.DEFAULT_HEIGHT)
        log.info(f"Window created: '{config.WINDOW_TITLE}'")

    #    FPS calculation 

    def _update_fps(self) -> None:
        """Recompute FPS once per second."""
        self._fps_counter += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._fps         = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start   = time.time()

    #  Keyboard handling 

    def _handle_key(self, key: int) -> None:
        """Respond to a single keypress."""

        #  Q / ESC → quit 
        if key in (ord("q"), ord("Q"), 27):
            self.running = False

        #  S → screenshot 
        elif key in (ord("s"), ord("S")):
            if self._current_frame is not None:
                path = self.capture.save_screenshot(self._current_frame.copy())
                if path:
                    self.renderer.show_notification("Screenshot Saved!")
                    log.info(f"Screenshot: {path}")

        #  E → toggle eye detection 
        elif key in (ord("e"), ord("E")):
            self.detect_eyes = not self.detect_eyes
            lbl = "ON" if self.detect_eyes else "OFF"
            self.renderer.show_notification(f"Eye Detection  {lbl}")
            log.info(f"Eye detection: {lbl}")

        #  M → toggle smile detection 
        elif key in (ord("m"), ord("M")):
            self.detect_smiles = not self.detect_smiles
            lbl = "ON" if self.detect_smiles else "OFF"
            self.renderer.show_notification(f"Smile Detection  {lbl}")
            log.info(f"Smile detection: {lbl}")

        #  C → toggle auto-capture 
        elif key in (ord("c"), ord("C")):
            active = self.capture.toggle_auto_capture()
            lbl = "ENABLED" if active else "DISABLED"
            self.renderer.show_notification(f"Auto-Capture  {lbl}")

        #  F → toggle fullscreen 
        elif key in (ord("f"), ord("F")):
            self.fullscreen = not self.fullscreen
            prop = (
                cv2.WINDOW_FULLSCREEN if self.fullscreen
                else cv2.WINDOW_NORMAL
            )
            cv2.setWindowProperty(
                config.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, prop
            )
            lbl = "ON" if self.fullscreen else "OFF"
            self.renderer.show_notification(f"Fullscreen  {lbl}")
            log.info(f"Fullscreen: {lbl}")

    #  Main loop 

    def run(self) -> None:
        """Initialise and start the detection loop."""
        self._init_camera()
        self._init_window()
        self.running = True
        log.info("Face Detection System running.  Press Q to quit.")

        try:
            while self.running:
                #  Frame capture 
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    log.warning("Empty frame received — skipping.")
                    continue

                # Mirror horizontally for natural selfie-view
                frame = cv2.flip(frame, 1)
                self._current_frame = frame

                #  Detection 
                detection = self.detector.detect(
                    frame,
                    detect_eyes   = self.detect_eyes,
                    detect_smiles = self.detect_smiles,
                )
                self._last_detection = detection

                #  Auto-capture tick 
                saved_paths = self.capture.update(frame, detection)
                if saved_paths:
                    n = len(saved_paths)
                    self.renderer.show_notification(
                        f"Auto-Saved  {n} Face{'s' if n > 1 else ''}!"
                    )

                #  FPS update 
                self._update_fps()

                #  Render UI overlays 
                state = {
                    "detect_eyes"  : self.detect_eyes,
                    "detect_smiles": self.detect_smiles,
                    "capture_mode" : self.capture.auto_mode,
                    "fullscreen"   : self.fullscreen,
                }
                self.renderer.render(frame, detection, self._fps, state)

                #  Display 
                cv2.imshow(config.WINDOW_TITLE, frame)

                #  Keyboard input (1 ms poll) 
                key = cv2.waitKey(1) & 0xFF
                if key != 255:          # 255 = no key pressed
                    self._handle_key(key)

        except KeyboardInterrupt:
            log.info("Stopped by keyboard interrupt (Ctrl+C).")

        except Exception as exc:
            log.exception(f"Unexpected runtime error: {exc}")

        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Release the camera and close all OpenCV windows."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        log.info(
            f"Shutdown complete.  "
            f"Total images saved: {self.capture.total_saved}"
        )


#  ENTRY POINT

def _print_banner() -> None:
    banner = r"""
 ╔══════════════════════════════════════════════════════════╗
 ║          AI-POWERED FACE DETECTION SYSTEM                ║
 ║          Powered by Python  ·  OpenCV  ·  Haar Cascade   ║
 
 ║  Controls                                                ║
 ║   Q / ESC  →  Quit                                       ║
 ║   S        →  Save screenshot                            ║
 ║   E        →  Toggle eye detection                       ║
 ║   M        →  Toggle smile detection                     ║
 ║   C        →  Toggle auto-capture  (saves every 5 s)    ║
 ║   F        →  Toggle fullscreen                          ║
 ╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def main() -> None:
    _print_banner()
    try:
        app = FaceDetectionApp()
        app.run()
    except RuntimeError as exc:
        log.error(str(exc))
        print(f"\n❌  Error: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        log.exception(f"Fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()