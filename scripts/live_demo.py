"""Live engagement demo — runs all AI modules and shows score on webcam.

Usage:
    python3 scripts/live_demo.py
Press Q to quit.
"""

import math
import os
import sys

# Ensure project root is on the path so 'src' can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

from src.config.scoring_weights import CourseType
from src.scoring.engagement_score import EngagementScorer

MODEL_PATH = "models/face_landmarker.task"

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# Correct MAR mouth indices: corners + upper/lower lip centres
MOUTH_TOP = 13      # upper inner lip centre
MOUTH_BOTTOM = 14   # lower inner lip centre
MOUTH_LEFT = 78     # left corner
MOUTH_RIGHT = 308   # right corner

EAR_THRESH = 0.22
MAR_THRESH = 0.35   # tighter threshold — mouth is open when > 0.35
DROWSY_FRAMES = 20  # ~1.3s at 15 FPS

# Key points to draw instead of all 478 (cleaner look)
KEY_LANDMARKS = (
    LEFT_EYE_IDX + RIGHT_EYE_IDX
    + [1, 4, 5, 195, 197]          # nose
    + [61, 291, 13, 14, 17, 0]     # mouth
    + [10, 338, 297, 332, 284,     # face contour
       251, 389, 356, 454, 323,
       361, 288, 397, 365, 379,
       378, 400, 377, 152, 148,
       176, 149, 150, 136, 172,
       58, 132, 93, 234, 127,
       162, 21, 54, 103, 67, 109]
)


def compute_ear(pts: list) -> float:
    """Eye Aspect Ratio — higher = more open."""
    A = math.dist((pts[1].x, pts[1].y), (pts[5].x, pts[5].y))
    B = math.dist((pts[2].x, pts[2].y), (pts[4].x, pts[4].y))
    C = math.dist((pts[0].x, pts[0].y), (pts[3].x, pts[3].y))
    return (A + B) / (2.0 * C + 1e-6)


def compute_mar(lm: list) -> float:
    """Mouth Aspect Ratio using inner lip + corner points."""
    vert = math.dist(
        (lm[MOUTH_TOP].x, lm[MOUTH_TOP].y),
        (lm[MOUTH_BOTTOM].x, lm[MOUTH_BOTTOM].y)
    )
    horiz = math.dist(
        (lm[MOUTH_LEFT].x, lm[MOUTH_LEFT].y),
        (lm[MOUTH_RIGHT].x, lm[MOUTH_RIGHT].y)
    )
    return vert / (horiz + 1e-6)


def run() -> None:
    scorer = EngagementScorer(CourseType.THEORY)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    drowsy_counter = 0
    yawn_counter = 0
    print("Demo running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        gaze_score = alertness_score = pose_score = expression_score = 50.0
        status_text = "No Face"
        status_color = (120, 120, 120)
        ear_val = mar_val = 0.0

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # ── Draw only key landmarks (cleaner) ────────────────────────────
            for idx in KEY_LANDMARKS:
                if idx < len(lm):
                    cx = int(lm[idx].x * w)
                    cy = int(lm[idx].y * h)
                    cv2.circle(frame, (cx, cy), 2, (0, 230, 100), -1)

            # Draw eye outlines
            for eye_idx in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
                pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_idx]
                pts_arr = np.array(pts, np.int32)
                cv2.polylines(frame, [pts_arr], True, (100, 255, 255), 1)

            # ── EAR Drowsiness ───────────────────────────────────────────────
            left_pts = [lm[i] for i in LEFT_EYE_IDX]
            right_pts = [lm[i] for i in RIGHT_EYE_IDX]
            ear_val = (compute_ear(left_pts) + compute_ear(right_pts)) / 2.0

            drowsy_counter = drowsy_counter + 1 if ear_val < EAR_THRESH else 0
            is_drowsy = drowsy_counter >= DROWSY_FRAMES
            alertness_score = 0.0 if is_drowsy else min(100.0, ear_val / 0.4 * 100)

            # ── MAR Yawn ─────────────────────────────────────────────────────
            mar_val = compute_mar(lm)
            yawn_counter = yawn_counter + 1 if mar_val > MAR_THRESH else 0
            is_yawning = yawn_counter >= 10
            if is_yawning:
                alertness_score = min(alertness_score, 50.0)

            # ── Head Pose (nose tip x-offset from centre) ────────────────────
            nose = lm[1]
            pose_score = max(0.0, 100.0 - abs(nose.x - 0.5) * 250)

            # ── Iris Gaze ────────────────────────────────────────────────────
            if len(lm) > 473:
                left_iris = lm[468]
                right_iris = lm[473]
                left_eye_cx = (lm[LEFT_EYE_IDX[0]].x + lm[LEFT_EYE_IDX[3]].x) / 2
                right_eye_cx = (lm[RIGHT_EYE_IDX[0]].x + lm[RIGHT_EYE_IDX[3]].x) / 2
                left_eye_w = abs(lm[LEFT_EYE_IDX[0]].x - lm[LEFT_EYE_IDX[3]].x)
                right_eye_w = abs(lm[RIGHT_EYE_IDX[0]].x - lm[RIGHT_EYE_IDX[3]].x)
                # Normalise iris offset by eye width: 0 = centred, 0.5 = at edge
                left_offset = abs(left_iris.x - left_eye_cx) / (left_eye_w + 1e-6)
                right_offset = abs(right_iris.x - right_eye_cx) / (right_eye_w + 1e-6)
                avg_offset = (left_offset + right_offset) / 2
                # avg_offset: ~0.05 = centre, ~0.4 = far side
                # Use 150 multiplier: 0.05*150=7.5 → score=92; 0.4*150=60 → score=40
                iris_score = max(0.0, 100.0 - avg_offset * 150)
                # Blend with pose (head direction) for robustness
                gaze_score = iris_score * 0.6 + pose_score * 0.4
            else:
                gaze_score = pose_score

            expression_score = 60.0  # placeholder (CNN not wired yet)

            if is_drowsy:
                status_text = "DROWSY!"
                status_color = (0, 50, 220)
            elif is_yawning:
                status_text = "Yawning"
                status_color = (0, 160, 255)
            else:
                status_text = "Engaged"
                status_color = (0, 200, 100)

        # ── Engagement score ──────────────────────────────────────────────────
        eng_score = scorer.compute_score(
            gaze=gaze_score,
            pose=pose_score,
            expression=expression_score,
            alertness=alertness_score,
        )

        bar_color = (
            (0, 220, 100) if eng_score >= 70
            else (0, 190, 255) if eng_score >= 40
            else (0, 50, 220)
        )

        # ── HUD ───────────────────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (220, 175), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "EngageIQ AI", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        cv2.putText(frame, f"EAR:   {ear_val:.3f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 150), 1)
        cv2.putText(frame, f"MAR:   {mar_val:.3f}", (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        cv2.putText(frame, f"Gaze:  {gaze_score:.0f}/100", (10, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 100), 1)
        cv2.putText(frame, f"Pose:  {pose_score:.0f}/100", (10, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 100), 1)
        cv2.putText(frame, f"Alert: {alertness_score:.0f}/100", (10, 138),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 100), 1)
        cv2.putText(frame, status_text, (10, 163),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

        # Score bar
        bar_w = int(eng_score / 100 * (w - 20))
        cv2.rectangle(frame, (10, h - 38), (w - 10, h - 14), (50, 50, 50), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (10, h - 38), (10 + bar_w, h - 14), bar_color, -1)
        cv2.putText(frame, f"Engagement Score: {eng_score:.0f} / 100",
                    (10, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)

        cv2.imshow("EngageIQ AI - Live Demo  [Q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    run()
