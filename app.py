# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
import threading
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
OUTPUT_FOLDER = os.path.join(os.getcwd(), "static")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

jobs = {}
jobs_lock = threading.Lock()

# ICC Law 21 — max elbow extension during delivery (degrees)
ICC_MAX_ELBOW_EXTENSION = 15.0

# ──────────────────────────────────────────────────────
# GEOMETRY HELPERS (2D + 3D)
# ──────────────────────────────────────────────────────

def joint_angle_2d(p1, p2, p3):
    """Interior angle at vertex p2 using normalized image coordinates."""
    a = np.array([p1.x, p1.y], dtype=np.float64)
    b = np.array([p2.x, p2.y], dtype=np.float64)
    c = np.array([p3.x, p3.y], dtype=np.float64)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cos_a = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def joint_angle_3d(p1, p2, p3):
    """Interior angle at vertex p2 using MediaPipe world landmarks (metres)."""
    a = np.array([p1.x, p1.y, p1.z], dtype=np.float64)
    b = np.array([p2.x, p2.y, p2.z], dtype=np.float64)
    c = np.array([p3.x, p3.y, p3.z], dtype=np.float64)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cos_a = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def line_angle_2d(p1, p2):
    """Angle of line p1→p2 relative to horizontal (degrees, 0 = level)."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return float(np.degrees(np.arctan2(-dy, dx)))


def upper_arm_horizontal_deg(shoulder, elbow):
    """How far upper arm deviates from horizontal (0° = perfectly level)."""
    return abs(line_angle_2d(shoulder, elbow))


def dist_2d(p1, p2):
    return float(np.hypot(p1.x - p2.x, p1.y - p2.y))


def dist_3d(p1, p2):
    return float(np.linalg.norm([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]))


def draw_angle_arc(frame, vertex, p1, p2, color, radius=32, thickness=2):
    h, w = frame.shape[:2]
    cx, cy = int(vertex.x * w), int(vertex.y * h)
    angle_a = float(np.degrees(np.arctan2(int(p1.y * h) - cy, int(p1.x * w) - cx)))
    angle_b = float(np.degrees(np.arctan2(int(p2.y * h) - cy, int(p2.x * w) - cx)))
    start_angle, end_angle = min(angle_a, angle_b), max(angle_a, angle_b)
    if end_angle - start_angle > 180:
        start_angle, end_angle = end_angle, start_angle + 360
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_angle, end_angle, color, thickness, cv2.LINE_AA)


def median_smooth(buf, val, window=7):
    buf.append(val)
    if len(buf) > window:
        buf.pop(0)
    return float(np.median(buf))


# ──────────────────────────────────────────────────────
# BOWLER TRACKING LOCK
# ──────────────────────────────────────────────────────

SCAN_FRAMES = 50
TOLERANCE = 0.30


class BowlerLock:
    def __init__(self):
        self.phase = "scan"
        self.history = []
        self.centre = None

    def feed(self, hip_x):
        self.history.append(hip_x)
        if len(self.history) >= SCAN_FRAMES:
            self.centre = float(np.median(self.history))
            self.phase = "locked"

    def accept(self, hip_x):
        if self.phase == "scan":
            return True
        return abs(hip_x - self.centre) <= TOLERANCE

    def update(self, hip_x):
        if self.phase == "locked" and self.centre is not None:
            self.centre = 0.93 * self.centre + 0.07 * hip_x


# ──────────────────────────────────────────────────────
# ICC ELBOW EXTENSION STATE MACHINE
# Measures extension from horizontal arm position to release — no scaling fudge.
# ──────────────────────────────────────────────────────

HORIZONTAL_TOL_DEG = 18.0       # upper arm within this of horizontal
WRIST_ABOVE_SHOULDER = 0.012    # normalized y margin
WRIST_DROP_RELEASE = 0.035      # wrist falls this much below peak → release
EA_SMOOTH = 7
DONE_HOLD_FRAMES_RATIO = 2.5    # seconds to hold result overlay


class ICCExtensionTracker:
    """
    ICC-compliant elbow extension measurement:
      1. Lock elbow angle when bowling-arm upper arm reaches horizontal.
      2. Track peak elbow angle through delivery until wrist drops past release.
      3. Extension = peak_angle - angle_at_horizontal (direct degrees, no multiplier).
    """

    def __init__(self, fps):
        self.state = "IDLE"
        self.angle_at_horizontal = None
        self.peak_elbow_angle = None
        self.min_wrist_y = 1.0
        self.extension_deg = 0.0
        self.release_elbow_angle = 0.0
        self.done_frames = 0
        self.done_hold = max(int(fps * DONE_HOLD_FRAMES_RATIO), 30)
        self.ea_buf = []
        self.peak_extension = 0.0
        self.peak_extension_frame_ext = 0.0
        self.peak_extension_release_angle = 0.0
        self.legal = True

    def reset_delivery(self):
        self.state = "IDLE"
        self.angle_at_horizontal = None
        self.peak_elbow_angle = None
        self.min_wrist_y = 1.0
        self.extension_deg = 0.0
        self.release_elbow_angle = 0.0
        self.done_frames = 0
        self.ea_buf.clear()

    def update(self, shoulder, elbow, wrist, elbow_angle_3d, elbow_angle_2d):
        """Returns (extension, release_angle, state, legal) for overlay."""
        ea = elbow_angle_3d if elbow_angle_3d is not None else elbow_angle_2d
        ea = median_smooth(self.ea_buf, ea, EA_SMOOTH)

        if not (50.0 <= ea <= 185.0):
            return self.extension_deg, self.release_elbow_angle, self.state, self.legal

        arm_horiz = upper_arm_horizontal_deg(shoulder, elbow) <= HORIZONTAL_TOL_DEG
        wrist_above = wrist.y < shoulder.y - WRIST_ABOVE_SHOULDER

        if self.state == "IDLE":
            if arm_horiz and wrist_above:
                self.angle_at_horizontal = ea
                self.peak_elbow_angle = ea
                self.min_wrist_y = wrist.y
                self.state = "DELIVERY"

        elif self.state == "DELIVERY":
            if wrist.y < self.min_wrist_y:
                self.min_wrist_y = wrist.y

            if ea > (self.peak_elbow_angle or 0):
                self.peak_elbow_angle = ea

            if self.angle_at_horizontal is not None:
                self.extension_deg = max(0.0, self.peak_elbow_angle - self.angle_at_horizontal)
                self.release_elbow_angle = self.peak_elbow_angle

            if wrist.y > self.min_wrist_y + WRIST_DROP_RELEASE:
                self.state = "DONE"
                self.done_frames = 0
                if self.extension_deg > self.peak_extension:
                    self.peak_extension = self.extension_deg
                    self.peak_extension_frame_ext = self.extension_deg
                    self.peak_extension_release_angle = self.release_elbow_angle
                self.legal = self.peak_extension <= ICC_MAX_ELBOW_EXTENSION

        elif self.state == "DONE":
            self.done_frames += 1
            if self.done_frames >= self.done_hold:
                self.reset_delivery()

        return self.extension_deg, self.release_elbow_angle, self.state, self.legal


# ──────────────────────────────────────────────────────
# DELIVERY PHASE DETECTOR (front/back leg, stride, release)
# ──────────────────────────────────────────────────────

class DeliveryAnalyzer:
    """Detects front-foot plant, identifies lead/trail leg, captures delivery metrics."""

    def __init__(self):
        self.phase = "run_up"
        self.run_up_start_hip_x = None
        self.run_up_end_hip_x = None
        self.front_foot = None          # 'left' or 'right'
        self.back_foot = None
        self.front_knee_at_delivery = []
        self.back_knee_at_delivery = []
        self.stride_lengths = []
        self.hip_angle_at_bfc = None    # back foot contact
        self.hip_angle_at_release = None
        self.shoulder_angles = []
        self.release_heights = []
        self.stride_count = 0
        self.foot_was_down = False
        self.delivery_stride_recorded = False
        self.body_height_ref = None

    def _hip_line_angle(self, l_hip, r_hip):
        return abs(line_angle_2d(l_hip, r_hip))

    def _shoulder_line_angle(self, l_sh, r_sh):
        """0° ≈ side-on, 90° ≈ chest-on (relative to camera)."""
        return abs(line_angle_2d(l_sh, r_sh))

    def _body_height(self, l_hip, r_hip, l_ankle, r_ankle):
        hip_y = (l_hip.y + r_hip.y) / 2.0
        ankle_y = max(l_ankle.y, r_ankle.y)
        return max(ankle_y - hip_y, 0.05)

    def feed_run_up(self, mid_hip_x):
        if self.run_up_start_hip_x is None:
            self.run_up_start_hip_x = mid_hip_x
        self.run_up_end_hip_x = mid_hip_x

    def feed_stride_counter(self, l_ankle, r_ankle):
        if max(l_ankle.y, r_ankle.y) > 0.82:
            if not self.foot_was_down:
                self.stride_count += 1
                self.foot_was_down = True
        else:
            self.foot_was_down = False

    def feed_frame(
        self,
        l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle,
        l_sh, r_sh, b_wri, icc_state,
    ):
        bh = self._body_height(l_hip, r_hip, l_ankle, r_ankle)
        if self.body_height_ref is None:
            self.body_height_ref = bh
        else:
            self.body_height_ref = 0.95 * self.body_height_ref + 0.05 * bh

        sh_angle = self._shoulder_line_angle(l_sh, r_sh)
        self.shoulder_angles.append(sh_angle)

        hip_angle = self._hip_line_angle(l_hip, r_hip)

        # Release height: wrist above ground, normalised to body height
        ground_y = max(l_ankle.y, r_ankle.y)
        rel_norm = (ground_y - b_wri.y) / self.body_height_ref
        self.release_heights.append(max(0.0, rel_norm) * 100.0)

        # Detect delivery: ICC tracker enters DELIVERY or DONE
        if icc_state in ("DELIVERY", "DONE") and not self.delivery_stride_recorded:
            self.delivery_stride_recorded = True
            self.phase = "delivery"

            # Front foot = ankle lower in frame (higher y) at plant
            if l_ankle.y > r_ankle.y + 0.02:
                self.front_foot, self.back_foot = "left", "right"
            elif r_ankle.y > l_ankle.y + 0.02:
                self.front_foot, self.back_foot = "right", "left"
            else:
                # fallback: leading ankle further in run-up direction
                self.front_foot = "left" if l_ankle.x > r_ankle.x else "right"
                self.back_foot = "right" if self.front_foot == "left" else "left"

            if self.hip_angle_at_bfc is None:
                self.hip_angle_at_bfc = hip_angle

            # Stride length = ankle separation / body height
            stride = dist_2d(l_ankle, r_ankle) / self.body_height_ref
            self.stride_lengths.append(stride)

        if icc_state == "DONE" and self.hip_angle_at_release is None:
            self.hip_angle_at_release = hip_angle

        # Collect knee angles during delivery window
        if icc_state in ("DELIVERY", "DONE"):
            if self.front_foot == "left":
                fk, bk = l_knee, r_knee
                fh, bh_ = l_hip, r_hip
                fa, ba = l_ankle, r_ankle
            elif self.front_foot == "right":
                fk, bk = r_knee, l_knee
                fh, bh_ = r_hip, l_hip
                fa, ba = r_ankle, l_ankle
            else:
                fk = bk = fh = bh_ = fa = ba = None

            if fk is not None:
                if fh.visibility > 0.5 and fk.visibility > 0.5 and fa.visibility > 0.5:
                    self.front_knee_at_delivery.append(joint_angle_2d(fh, fk, fa))
                if bh_.visibility > 0.5 and bk.visibility > 0.5 and ba.visibility > 0.5:
                    self.back_knee_at_delivery.append(joint_angle_2d(bh_, bk, ba))

    def summary(self):
        run_up = 0.0
        if self.run_up_start_hip_x is not None and self.run_up_end_hip_x is not None:
            run_up = abs(self.run_up_end_hip_x - self.run_up_start_hip_x)
            if self.body_height_ref:
                run_up = run_up / self.body_height_ref

        hip_rot = 0.0
        if self.hip_angle_at_bfc is not None and self.hip_angle_at_release is not None:
            hip_rot = abs(self.hip_angle_at_release - self.hip_angle_at_bfc)

        avg_sh = float(np.mean(self.shoulder_angles)) if self.shoulder_angles else 0.0

        return {
            "strides": self.stride_count,
            "avg_l_knee": int(np.mean(self.front_knee_at_delivery)) if self.front_knee_at_delivery else (
                int(np.mean(self.back_knee_at_delivery)) if self.back_knee_at_delivery else 0
            ),
            "avg_r_knee": int(np.mean(self.back_knee_at_delivery)) if self.back_knee_at_delivery else 0,
            "front_knee": int(np.mean(self.front_knee_at_delivery)) if self.front_knee_at_delivery else 0,
            "back_knee": int(np.mean(self.back_knee_at_delivery)) if self.back_knee_at_delivery else 0,
            "shoulder_alignment": round(avg_sh, 1),
            "hip_rotation": round(hip_rot, 1),
            "avg_release_score": round(float(np.percentile(self.release_heights, 90)), 1) if self.release_heights else 0,
            "release_height_pct": round(float(np.percentile(self.release_heights, 90)), 1) if self.release_heights else 0,
            "run_up_length": round(run_up * 100, 1),
            "stride_length": round(float(np.mean(self.stride_lengths)) * 100, 1) if self.stride_lengths else 0,
            "avg_velocity": 0,
        }


# ──────────────────────────────────────────────────────
# VIDEO RE-ENCODE
# ──────────────────────────────────────────────────────

def reencode_for_web(raw, final):
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-movflags", "+faststart", "-an", final,
            ],
            capture_output=True,
            timeout=180,
        )
        if os.path.exists(raw):
            os.remove(raw)
        return r.returncode == 0
    except Exception as ex:
        print(f"Re-encode error: {ex}")
        if os.path.exists(raw):
            try:
                os.rename(raw, final)
            except Exception:
                pass
        return False


# ──────────────────────────────────────────────────────
# MAIN PROCESSOR
# ──────────────────────────────────────────────────────

def process_bowling_video(video_path, output_path, job_id):
    pose = cap = out = None
    raw_out = output_path.replace(".mp4", "_raw.mp4")

    l_knee_angs, r_knee_angs, velocities = [], [], []
    peak_icc_extension = 0.0
    peak_release_angle = 0.0
    icc_legal = True

    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        tpf = 1.0 / fps

        scale = min(1.0, 720 / max(orig_w, orig_h))
        proc_w = int(orig_w * scale)
        proc_h = int(orig_h * scale)

        out = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (proc_w, proc_h))

        prev_hip_x = None
        bowling_side = None
        lock = BowlerLock()
        icc = ICCExtensionTracker(fps)
        delivery = DeliveryAnalyzer()

        sf = max(min(proc_w, proc_h) / 720 * 0.52, 0.30)
        fth = max(1, int(sf * 2))
        pad = max(6, int(sf * 10))
        line_h = max(22, int(proc_h * 0.055))
        VIS = 0.45

        def put_label(img, text, x, y, color):
            fnt = cv2.FONT_HERSHEY_SIMPLEX
            tw, txh = cv2.getTextSize(text, fnt, sf, fth)[0]
            bl_b = cv2.getTextSize(text, fnt, sf, fth)[1]
            x = max(pad, min(x, proc_w - tw - pad * 2))
            y = max(txh + pad, min(y, proc_h - pad))
            ov = img.copy()
            cv2.rectangle(ov, (x - pad, y - txh - pad), (x + tw + pad, y + bl_b + pad // 2), (10, 14, 22), -1)
            cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
            cv2.putText(img, text, (x, y), fnt, sf, color, fth, cv2.LINE_AA)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if scale != 1.0:
                frame = cv2.resize(frame, (proc_w, proc_h))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                out.write(frame)
                continue

            lm = res.pose_landmarks.landmark
            wlm = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None

            l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
            mid_hip_x = (l_hip.x + r_hip.x) / 2.0

            if lock.phase == "scan":
                if l_hip.visibility > 0.4 or r_hip.visibility > 0.4:
                    lock.feed(mid_hip_x)

            if not lock.accept(mid_hip_x):
                out.write(frame)
                continue

            lock.update(mid_hip_x)
            delivery.feed_run_up(mid_hip_x)

            mp_drawing.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
            )

            l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
            l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_elb = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            r_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            l_wri = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wri = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Continuous L/R knee tracking (all frames)
            if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                la = joint_angle_2d(l_hip, l_knee, l_ankle)
                l_knee_angs.append(la)
                col = (0, 255, 0) if la >= 165 else ((0, 200, 255) if la >= 140 else (232, 234, 242))
                draw_angle_arc(frame, l_knee, l_hip, l_ankle, col, radius=32)
                put_label(frame, f"L Knee: {int(la)} deg", int(l_knee.x * proc_w) + 14, int(l_knee.y * proc_h), col)

            if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                ra = joint_angle_2d(r_hip, r_knee, r_ankle)
                r_knee_angs.append(ra)
                col = (0, 255, 0) if ra >= 165 else ((0, 200, 255) if ra >= 140 else (232, 234, 242))
                draw_angle_arc(frame, r_knee, r_hip, r_ankle, col, radius=32)
                put_label(frame, f"R Knee: {int(ra)} deg", int(r_knee.x * proc_w) + 14, int(r_knee.y * proc_h) - 24, col)

            # Detect bowling arm
            b_sh = b_elb = b_wri = None
            if (
                l_sh.visibility > VIS and r_sh.visibility > VIS
                and l_elb.visibility > VIS and r_elbow.visibility > VIS
                and l_wri.visibility > VIS and r_wri.visibility > VIS
            ):
                if bowling_side is None:
                    l_lift = l_sh.y - l_wri.y
                    r_lift = r_sh.y - r_wri.y
                    if abs(l_lift - r_lift) > 0.04:
                        bowling_side = "left" if l_lift > r_lift else "right"

                if bowling_side is not None:
                    b_sh = l_sh if bowling_side == "left" else r_sh
                    b_elb = l_elb if bowling_side == "left" else r_elbow
                    b_wri = l_wri if bowling_side == "left" else r_wri

                    ea_2d = joint_angle_2d(b_sh, b_elb, b_wri)
                    ea_3d = None
                    if wlm is not None:
                        ws = wlm[mp_pose.PoseLandmark.LEFT_SHOULDER if bowling_side == "left" else mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        we = wlm[mp_pose.PoseLandmark.LEFT_ELBOW if bowling_side == "left" else mp_pose.PoseLandmark.RIGHT_ELBOW]
                        ww = wlm[mp_pose.PoseLandmark.LEFT_WRIST if bowling_side == "left" else mp_pose.PoseLandmark.RIGHT_WRIST]
                        ea_3d = joint_angle_3d(ws, we, ww)

                    ext, rel_ang, icc_state, legal = icc.update(b_sh, b_elb, b_wri, ea_3d, ea_2d)

                    if icc.peak_extension > peak_icc_extension:
                        peak_icc_extension = icc.peak_extension
                        peak_release_angle = icc.peak_extension_release_angle
                        icc_legal = icc.legal

                    delivery.feed_frame(
                        l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle,
                        l_sh, r_sh, b_wri, icc_state,
                    )

                    draw_angle_arc(frame, b_elb, b_sh, b_wri, (0, 242, 254), radius=30)
                    wx = int(b_wri.x * proc_w)
                    wy = int(b_wri.y * proc_h)

                    if icc_state in ("DELIVERY", "DONE"):
                        if ext <= ICC_MAX_ELBOW_EXTENSION:
                            col_status = (0, 255, 0)
                            legal_lbl = f"ICC Ext: {ext:.1f} deg (LEGAL <=15)"
                        else:
                            col_status = (0, 0, 255)
                            legal_lbl = f"ICC Ext: {ext:.1f} deg (ILLEGAL >15)"
                        put_label(frame, legal_lbl, wx + 14, wy - 44, col_status)
                        put_label(frame, f"Elbow at release: {int(rel_ang)} deg", wx + 14, wy - 22, (232, 234, 242))
                        if icc.angle_at_horizontal is not None:
                            put_label(
                                frame,
                                f"At horizontal: {int(icc.angle_at_horizontal)} deg",
                                wx + 14, wy - 2, (180, 180, 180),
                            )
                    else:
                        put_label(frame, f"Elbow: {int(ea_3d or ea_2d)} deg", wx + 14, wy - 8, (200, 200, 200))

            delivery.feed_stride_counter(l_ankle, r_ankle)
            put_label(frame, f"Strides: {delivery.stride_count}", 16, line_h, (255, 255, 255))

            # Shoulder alignment overlay
            if l_sh.visibility > 0.5 and r_sh.visibility > 0.5:
                sh_a = delivery._shoulder_line_angle(l_sh, r_sh)
                chest_pct = min(100, round(sh_a / 90.0 * 100))
                put_label(frame, f"Shoulder: {int(sh_a)} deg ({chest_pct}% chest-on)", 16, line_h * 3 + 8, (245, 230, 66))

            # Hip rotation overlay during delivery
            if delivery.hip_angle_at_bfc is not None:
                cur_hip = delivery._hip_line_angle(l_hip, r_hip)
                rot = abs(cur_hip - delivery.hip_angle_at_bfc)
                put_label(frame, f"Hip rot: {rot:.1f} deg", 16, line_h * 4 + 12, (0, 242, 254))

            if l_hip.visibility > 0.4:
                cx = l_hip.x * proc_w
                if prev_hip_x is not None:
                    vel = abs(cx - prev_hip_x) / tpf
                    velocities.append(vel)
                    put_label(frame, f"Vel: {int(vel)} px/s", 16, line_h * 2 + 4, (0, 242, 254))
                prev_hip_x = cx

            frame = cv2.convertScaleAbs(frame, alpha=1.02, beta=1)
            out.write(frame)

        cap.release()
        cap = None
        out.release()
        out = None
        pose.close()
        pose = None

    except Exception as exc:
        print(f"Processing error: {exc}")
        with jobs_lock:
            jobs[job_id] = {"status": "error", "error": str(exc)}
        return

    reencode_for_web(raw_out, output_path)

    summary = delivery.summary()
    summary["avg_l_knee"] = int(np.mean(l_knee_angs)) if l_knee_angs else summary.get("front_knee", 0)
    summary["avg_r_knee"] = int(np.mean(r_knee_angs)) if r_knee_angs else summary.get("back_knee", 0)
    summary["avg_velocity"] = int(np.mean(velocities)) if velocities else 0
    summary["icc_elbow_extension"] = round(peak_icc_extension, 1)
    summary["icc_legal"] = icc_legal
    summary["release_elbow_angle"] = round(peak_release_angle, 1)
    summary["bowling_arm"] = bowling_side or "unknown"

    with jobs_lock:
        jobs[job_id] = {
            "status": "done",
            "video_url": f"/static/{os.path.basename(output_path)}",
            "summary": summary,
        }


# ──────────────────────────────────────────────────────
# FLASK ROUTES
# ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    job_id = str(uuid.uuid4())[:8]
    input_path = os.path.join(UPLOAD_FOLDER, f"input_{job_id}.mp4")
    output_name = f"analyzed_{job_id}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    file.save(input_path)

    for f in os.listdir(OUTPUT_FOLDER):
        if f.startswith("analyzed_") and f != output_name:
            try:
                os.remove(os.path.join(OUTPUT_FOLDER, f))
            except Exception:
                pass

    with jobs_lock:
        jobs[job_id] = {"status": "processing"}

    threading.Thread(
        target=process_bowling_video,
        args=(input_path, output_path, job_id),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id}), 202


@app.route("/status/<job_id>")
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)


@app.route("/static/<filename>")
def serve_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
