import cv2
import numpy as np
from typing import Dict

import config
from detector import DetectionResult


class UIRenderer:
  
    def __init__(self) -> None:
        self._notif_text  : str = ""
        self._notif_timer : int = 0

    #  Public API 

    def show_notification(self, text: str) -> None:
        """Display a centred notification message for a short time."""
        self._notif_text  = text
        self._notif_timer = config.NOTIFICATION_DURATION

    def render(
        self,
        frame      : np.ndarray,
        detection  : DetectionResult,
        fps        : float,
        state      : dict,
    ) -> None:
        
        h, w = frame.shape[:2]

        self._draw_top_bar(frame, fps, detection, state, w)
        self._draw_face_boxes(frame, detection)
        self._draw_sub_boxes(frame, detection)
        self._draw_bottom_bar(frame, w, h)

        if self._notif_timer > 0:
            self._draw_notification(frame, w, h)
            self._notif_timer -= 1

    #  Semi-transparent rectangle helper 

    @staticmethod
    def _overlay_rect(
        frame : np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color : tuple,
        alpha : float = 0.80,
    ) -> None:
        """
        Paint a semi-transparent filled rectangle.
        alpha=1.0 → fully opaque | alpha=0.0 → fully transparent.
        """
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        bg  = np.zeros_like(roi, dtype=np.uint8)
        bg[:] = color                                    # fill with solid colour
        cv2.addWeighted(bg, alpha, roi, 1.0 - alpha, 0, roi)
        frame[y1:y2, x1:x2] = roi

    #  Corner-style bounding box 

    @staticmethod
    def _draw_corner_box(
        frame     : np.ndarray,
        x: int, y: int, w: int, h: int,
        color     : tuple,
        thickness : int = 2,
        corner_len: int = 22,
    ) -> None:
       
        corners = [
            # (origin, horizontal end, vertical end)
            ((x,   y),   (x + corner_len,   y),   (x,   y + corner_len)),
            ((x+w, y),   (x+w - corner_len, y),   (x+w, y + corner_len)),
            ((x,   y+h), (x + corner_len,   y+h), (x,   y+h - corner_len)),
            ((x+w, y+h), (x+w - corner_len, y+h), (x+w, y+h - corner_len)),
        ]
        for origin, h_end, v_end in corners:
            cv2.line(frame, origin, h_end, color, thickness, cv2.LINE_AA)
            cv2.line(frame, origin, v_end, color, thickness, cv2.LINE_AA)

    #    Top status bar 

    def _draw_top_bar(
        self,
        frame     : np.ndarray,
        fps       : float,
        detection : DetectionResult,
        state     : dict,
        w         : int,
    ) -> None:
        BAR_H = 52
        self._overlay_rect(frame, 0, 0, w, BAR_H, config.CLR_DARK_BG, alpha=0.85)
        cv2.line(frame, (0, BAR_H), (w, BAR_H), (55, 55, 55), 1)

        #  Title 
        cv2.putText(
            frame, "AI FACE DETECTION",
            (14, 33), config.FONT_BOLD, config.FONT_MD,
            config.CLR_ACCENT, 1, cv2.LINE_AA,
        )

        #  FPS (colour-coded by speed) 
        fps_clr = (
            config.CLR_FPS_GOOD  if fps >= 20 else
            config.CLR_FPS_OK    if fps >= 10 else
            config.CLR_FPS_LOW
        )
        cv2.putText(
            frame, f"FPS: {fps:5.1f}",
            (232, 33), config.FONT, config.FONT_MD, fps_clr, 1, cv2.LINE_AA,
        )

        #  Face count 
        fc      = len(detection.faces)
        fc_clr  = config.CLR_FACE_BOX if fc > 0 else (120, 120, 120)
        cv2.putText(
            frame, f"Faces: {fc}",
            (368, 33), config.FONT, config.FONT_MD, fc_clr, 1, cv2.LINE_AA,
        )

        #  Mode indicators (dot + label) 
        mode_items = [
            ("EYES",    "detect_eyes",   config.CLR_EYE_BOX,   490),
            ("SMILE",   "detect_smiles", config.CLR_SMILE_BOX, 590),
            ("CAPTURE", "capture_mode",  config.CLR_CAPTURE,   698),
        ]
        for label, key, clr, xpos in mode_items:
            active  = state.get(key, False)
            dot_c   = clr            if active else (55, 55, 55)
            txt_c   = clr            if active else (75, 75, 75)
            status  = "ON"           if active else "OFF"
            cv2.circle(frame, (xpos, 28), 5, dot_c, -1, cv2.LINE_AA)
            cv2.putText(
                frame, f"{label}: {status}",
                (xpos + 11, 33), config.FONT, config.FONT_SM,
                txt_c, 1, cv2.LINE_AA,
            )

        #  LIVE pulse dot 
        live_clr = config.CLR_FACE_BOX if fc > 0 else (70, 70, 70)
        cv2.circle(frame, (w - 22, 26), 7, live_clr, -1, cv2.LINE_AA)
        cv2.putText(
            frame, "LIVE",
            (w - 60, 33), config.FONT, config.FONT_SM,
            live_clr, 1, cv2.LINE_AA,
        )

    #  Face bounding boxes 

    def _draw_face_boxes(
        self, frame: np.ndarray, detection: DetectionResult
    ) -> None:
        """Draw corner-bracket box + ID label + confidence for each face."""
        # Build reverse map: bounding rect → face ID
        rect_to_id: Dict = {
            tuple(v): k for k, v in detection.face_ids.items()
        }

        for rect in detection.faces:
            x, y, w, h = rect
            fid  = rect_to_id.get(tuple(rect), "?")
            conf = self._compute_conf(rect, frame.shape)

            # Faint full rectangle (depth effect behind corner brackets)
            dim_clr = tuple(max(0, c // 4) for c in config.CLR_FACE_BOX)
            cv2.rectangle(frame, (x, y), (x + w, y + h), dim_clr, 1)

            # Corner bracket overlay
            self._draw_corner_box(
                frame, x, y, w, h, config.CLR_FACE_BOX, thickness=2, corner_len=22
            )

            #  ID + confidence label 
            label = f"  ID:{fid}  {conf}%  "
            (tw, th), bl = cv2.getTextSize(
                label, config.FONT_BOLD, config.FONT_SM, 1
            )

            # Place above face; shift down if too close to top edge
            label_y = y - 8 if y > th + 14 else y + h + th + 10

            self._overlay_rect(
                frame,
                max(0, x - 2),
                max(0, label_y - th - 4),
                min(frame.shape[1], x + tw + 2),
                label_y + bl + 2,
                config.CLR_DARK_BG, alpha=0.72,
            )
            cv2.putText(
                frame, label,
                (x, label_y), config.FONT_BOLD, config.FONT_SM,
                config.CLR_ID_TAG, 1, cv2.LINE_AA,
            )

            #  "Smiling" tag if smile detected inside this face 
            if detection.smiles:
                for (sx, sy, sw, sh) in detection.smiles:
                    if x <= sx <= x + w and y <= sy <= y + h:
                        self._draw_label_tag(
                            frame, x + w - 2, y + h - 2,
                            "Smiling :)", config.CLR_SMILE_BOX
                        )
                        break

    #  Eye & smile rectangles 

    @staticmethod
    def _draw_sub_boxes(
        frame: np.ndarray, detection: DetectionResult
    ) -> None:
        for (x, y, w, h) in detection.eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          config.CLR_EYE_BOX, 1)

        for (x, y, w, h) in detection.smiles:
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          config.CLR_SMILE_BOX, 1)

    #  Bottom hints bar 

    def _draw_bottom_bar(
        self, frame: np.ndarray, w: int, h: int
    ) -> None:
        BAR_H = 30
        self._overlay_rect(frame, 0, h - BAR_H, w, h, config.CLR_DARK_BG, alpha=0.82)
        cv2.line(frame, (0, h - BAR_H), (w, h - BAR_H), (55, 55, 55), 1)

        hints = (
            "  Q / ESC : Quit     "
            "S : Screenshot     "
            "E : Toggle Eyes     "
            "M : Toggle Smile     "
            "C : Auto-Capture     "
            "F : Fullscreen"
        )
        cv2.putText(
            frame, hints,
            (8, h - 9), config.FONT, config.FONT_SM - 0.07,
            (145, 145, 145), 1, cv2.LINE_AA,
        )

    #  Centre notification 

    def _draw_notification(
        self, frame: np.ndarray, w: int, h: int
    ) -> None:
        """Show a fading centred message for screenshot / capture events."""
        (tw, th), _ = cv2.getTextSize(
            self._notif_text, config.FONT_BOLD, config.FONT_LG, 2
        )
        pad = 22
        bx1 = (w - tw) // 2 - pad
        bx2 = (w + tw) // 2 + pad
        by1 = h // 2 - th - pad
        by2 = h // 2 + pad

        self._overlay_rect(frame, bx1, by1, bx2, by2, config.CLR_DARK_BG, alpha=0.88)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), config.CLR_CAPTURE, 1)

        tx = (w - tw) // 2
        ty = h // 2 + th // 2
        cv2.putText(
            frame, self._notif_text,
            (tx, ty), config.FONT_BOLD, config.FONT_LG,
            config.CLR_CAPTURE, 2, cv2.LINE_AA,
        )

    #  Misc helpers 

    @staticmethod
    def _compute_conf(rect, frame_shape) -> float:
        x, y, w, h   = rect
        fh, fw        = frame_shape[:2]
        size_score    = min(1.0, (w * h) / max(1, fw * fh * 0.30)) * 60
        cx, cy        = x + w / 2, y + h / 2
        dist          = ((cx - fw / 2) ** 2 + (cy - fh / 2) ** 2) ** 0.5
        max_dist      = ((fw / 2) ** 2 + (fh / 2) ** 2) ** 0.5
        return round(min(100.0, size_score + (1.0 - dist / max_dist) * 40), 1)

    @staticmethod
    def _draw_label_tag(
        frame : np.ndarray,
        x     : int,
        y     : int,
        text  : str,
        color : tuple,
    ) -> None:
        """Draw a small coloured pill-label at (x, y)."""
        (tw, th), bl = cv2.getTextSize(text, config.FONT, config.FONT_SM, 1)
        x1, y1 = x - tw - 8, y - th - 6
        x2, y2 = x + 4, y + bl + 2
        # Semi-transparent background
        UIRenderer._overlay_rect(
            frame, x1, y1, x2, y2, (0, 0, 0), alpha=0.70
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            frame, text,
            (x - tw - 4, y), config.FONT, config.FONT_SM,
            color, 1, cv2.LINE_AA,
        )