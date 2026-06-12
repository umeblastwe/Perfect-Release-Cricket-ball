# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
import threading
import uuid
from collections import deque
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER']      = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

jobs      = {}
jobs_lock = threading.Lock()

# ──────────────────────────────────────────────────────
# MATHS
# ──────────────────────────────────────────────────────

def joint_angle_2d(p1, p2, p3):
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def elbow_angle_3d(sh, elb, wri):
    s = np.array([sh.x,  sh.y,  sh.z])
    e = np.array([elb.x, elb.y, elb.z])
    w = np.array([wri.x, wri.y, wri.z])
    vse, vew = s - e, w - e
    cos = np.dot(vse, vew) / (np.linalg.norm(vse) * np.linalg.norm(vew) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def draw_arc(frame, vtx, p1, p2, color, radius=32, thick=2):
    h, w = frame.shape[:2]
    cx, cy = int(vtx.x * w), int(vtx.y * h)
    ang_a = float(np.degrees(np.arctan2(int(p1.y*h)-cy, int(p1.x*w)-cx)))
    ang_b = float(np.degrees(np.arctan2(int(p2.y*h)-cy, int(p2.x*w)-cx)))
    sa, ea = min(ang_a, ang_b), max(ang_a, ang_b)
    if ea - sa > 180:
        sa, ea = ea, sa + 360
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, sa, ea, color, thick, cv2.LINE_AA)


# ──────────────────────────────────────────────────────
# BOWLER LOCK
# ──────────────────────────────────────────────────────
# HOW IT WORKS:
#   For the first SCAN_FRAMES frames we run MediaPipe on the full frame and
#   record where we detect hips.  Then we compute the median position and lock
#   onto that person.  After locking, any detection whose mid-hip X is more
#   than TOLERANCE away from the locked centre is rejected — that's the umpire
#   or batsman.  The lock centre drifts slowly (EMA) so a sprinting bowler
#   who runs across the frame stays tracked.

SCAN_FRAMES = 50        # frames spent in identification phase
TOLERANCE   = 0.30      # normalised-X units of allowed drift from lock centre


class BowlerLock:
    def __init__(self):
        self.phase     = 'scan'      # 'scan' → 'locked'
        self.history   = []
        self.centre    = None

    def feed(self, hip_x):
        """Call with mid-hip X (0-1) every frame during scan phase."""
        self.history.append(hip_x)
        if len(self.history) >= SCAN_FRAMES:
            self.centre = float(np.median(self.history))
            self.phase  = 'locked'

    def accept(self, hip_x):
        """Returns True if this detection is the bowler."""
        if self.phase == 'scan':
            # During scan always accept so we gather data
            return True
        return abs(hip_x - self.centre) <= TOLERANCE

    def update(self, hip_x):
        """Slow drift so we track a bowler moving across frame."""
        if self.phase == 'locked' and self.centre is not None:
            self.centre = 0.93 * self.centre + 0.07 * hip_x


# ──────────────────────────────────────────────────────
# FFMPEG
# ──────────────────────────────────────────────────────

def reencode_for_web(raw, final):
    try:
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', raw,
             '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
             '-movflags', '+faststart', '-an', final],
            capture_output=True, timeout=180)
        if os.path.exists(raw):
            os.remove(raw)
        return r.returncode == 0
    except Exception as ex:
        print(f"ffmpeg: {ex}")
        if os.path.exists(raw):
            try: os.rename(raw, final)
            except Exception: pass
        return False


# ──────────────────────────────────────────────────────
# CORE VIDEO PROCESSOR
# ──────────────────────────────────────────────────────

def process_bowling_video(video_path, output_path, job_id):
    pose = cap = out = None
    try:
        mp_pose    = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        tpf    = 1.0 / fps

        scale  = min(1.0, 720 / max(orig_w, orig_h))
        proc_w = int(orig_w * scale)
        proc_h = int(orig_h * scale)

        raw_out = output_path.replace('.mp4', '_raw.mp4')
        out = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (proc_w, proc_h))
        if not out.isOpened():
            raise RuntimeError("Cannot create writer")

        # ── stats ──
        prev_hip_x    = None
        stride_count  = 0
        foot_was_down = False
        l_knee_angs   = []
        r_knee_angs   = []
        rel_scores    = []
        velocities    = []

        # ── elbow / ICC extension state machine ──
        # ICC rule: extension = angle_at_release − angle_at_horizontal
        # One delivery cycle: IDLE → HORIZ → HUNT → DONE → IDLE
        bowling_side    = None    # 'left' | 'right', locked on first confident frame
        del_state       = 'IDLE'  # delivery state machine
        theta_h         = None    # elbow angle when arm first hits horizontal
        min_wrist_y     = 1.0     # track highest wrist position (smallest y = highest)
        ea_at_peak      = None    # elbow angle at that highest wrist moment (= release)
        display_ext     = 0.0     # extension shown on overlay (updated per delivery)
        display_ea      = 0.0     # elbow angle shown on overlay

        # ── bowler lock ──
        lock = BowlerLock()

        # ── font scale (computed once) ──
        sf  = max(min(proc_w, proc_h) / 720 * 0.52, 0.30)
        fth = max(1, int(sf * 2))
        pad = max(6, int(sf * 10))

        def put_label(img, text, x, y, color):
            fnt = cv2.FONT_HERSHEY_SIMPLEX
            (tw, txh), bl_b = cv2.getTextSize(text, fnt, sf, fth)
            x = max(pad, min(x, proc_w - tw - pad * 2))
            y = max(txh + pad, min(y, proc_h - pad))
            ov = img.copy()
            cv2.rectangle(ov,
                          (x - pad,       y - txh - pad),
                          (x + tw + pad,  y + bl_b + pad // 2),
                          (10, 14, 22), -1)
            cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
            cv2.putText(img, text, (x, y), fnt, sf, color, fth, cv2.LINE_AA)

        # ── main loop ──
        while True:
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

            lm   = res.pose_landmarks.landmark
            l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            # Mid-hip X used for bowler identity
            mid_hip_x = (l_hip.x + r_hip.x) / 2.0

            # Feed lock system
            if lock.phase == 'scan':
                if l_hip.visibility > 0.4 or r_hip.visibility > 0.4:
                    lock.feed(mid_hip_x)

            # Reject umpire / batsman
            if not lock.accept(mid_hip_x):
                out.write(frame)
                continue

            # Update drift
            lock.update(mid_hip_x)

            # ── Draw skeleton on BOWLER only ──
            mp_drawing.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
            )

            l_knee  = lm[mp_pose.PoseLandmark.LEFT_KNEE]
            l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            r_knee  = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
            r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            l_sh    = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh    = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_elb   = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            r_elb   = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            l_wri   = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wri   = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

            line_h = max(22, int(proc_h * 0.055))

            # ── LEFT KNEE ──
            if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                la  = joint_angle_2d(l_hip, l_knee, l_ankle)
                l_knee_angs.append(la)
                col = (0,255,0) if la >= 165 else ((0,200,255) if la >= 140 else (232,234,242))
                draw_arc(frame, l_knee, l_hip, l_ankle, col, radius=32)
                put_label(frame, f"L Knee: {int(la)} deg",
                          int(l_knee.x*proc_w)+14, int(l_knee.y*proc_h), col)

            # ── RIGHT KNEE ──
            if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                ra  = joint_angle_2d(r_hip, r_knee, r_ankle)
                r_knee_angs.append(ra)
                col = (0,255,0) if ra >= 165 else ((0,200,255) if ra >= 140 else (232,234,242))
                draw_arc(frame, r_knee, r_hip, r_ankle, col, radius=32)
                put_label(frame, f"R Knee: {int(ra)} deg",
                          int(r_knee.x*proc_w)+14, int(r_knee.y*proc_h)-24, col)

            # ── ELBOW EXTENSION  (ICC correct measurement) ──
            # The 15-degree rule measures straightening between horizontal and release.
            # We detect: IDLE → arm below shoulder
            #            HORIZ → arm reaches horizontal, snapshot elbow angle (theta_h)
            #            HUNT  → track wrist until it peaks (highest y = release moment)
            #            DONE  → compute extension = ea_at_peak - theta_h, display it
            if l_wri.visibility > 0.4 and r_wri.visibility > 0.4:

                # Lock bowling side once: whichever wrist is clearly above the shoulder
                if bowling_side is None:
                    l_above = l_sh.y - l_wri.y   # positive = wrist above shoulder
                    r_above = r_sh.y - r_wri.y
                    if max(l_above, r_above) > 0.05:
                        bowling_side = 'left' if l_above > r_above else 'right'

                if bowling_side is not None:
                    b_sh  = l_sh  if bowling_side == 'left' else r_sh
                    b_elb = l_elb if bowling_side == 'left' else r_elb
                    b_wri = l_wri if bowling_side == 'left' else r_wri

                    ea_now = elbow_angle_3d(b_sh, b_elb, b_wri)

                    # ── STATE MACHINE ──
                    wrist_above_sh  = b_sh.y - b_wri.y          # positive = wrist above sh
                    elbow_near_sh_h = abs(b_sh.y - b_elb.y)     # small = arm horizontal

                    if del_state == 'IDLE':
                        # Wait for arm to reach roughly horizontal (elbow at shoulder height)
                        # and wrist still rising (wrist above shoulder)
                        if elbow_near_sh_h < 0.12 and wrist_above_sh > -0.05:
                            theta_h     = ea_now
                            min_wrist_y = b_wri.y   # initialise peak tracker
                            ea_at_peak  = ea_now
                            del_state   = 'HUNT'

                    elif del_state == 'HUNT':
                        # Track the wrist as it continues to rise toward release
                        # Release = highest wrist position (minimum Y value)
                        if b_wri.y < min_wrist_y:
                            min_wrist_y = b_wri.y
                            ea_at_peak  = ea_now

                        # Wrist has started falling — release has passed, compute extension
                        if b_wri.y > min_wrist_y + 0.04:
                            if theta_h is not None and ea_at_peak is not None:
                                raw_ext = ea_at_peak - theta_h
                                # Extension means the arm STRAIGHTENED (angle grew toward 180)
                                display_ext = max(0.0, raw_ext)
                                display_ea  = ea_at_peak
                            del_state = 'DONE'

                    elif del_state == 'DONE':
                        # Hold display until arm drops back below shoulder → new delivery
                        if b_wri.y > b_sh.y + 0.15:
                            del_state    = 'IDLE'
                            theta_h      = None
                            min_wrist_y  = 1.0
                            ea_at_peak   = None

                    # ── OVERLAY ──
                    wx = int(b_wri.x * proc_w)
                    wy = int(b_wri.y * proc_h)
                    draw_arc(frame, b_elb, b_sh, b_wri, (0,242,254), radius=28)

                    if del_state in ('HUNT', 'DONE'):
                        ext_color = (0,255,0) if display_ext <= 15 else (0,200,255)
                        put_label(frame, f"Elbow Ext: {display_ext:.1f} deg",  wx+14, wy-32, ext_color)
                        put_label(frame, f"Elbow Angle: {int(display_ea)} deg", wx+14, wy-6,  (200,200,200))
                    else:
                        # During IDLE show live angle only (no extension yet)
                        put_label(frame, f"Elbow Angle: {int(ea_now)} deg", wx+14, wy-6, (200,200,200))

                    # Release height score (wrist vs ground)
                    ground = max(l_ankle.y, r_ankle.y)
                    rel_scores.append((ground - b_wri.y) * 100)

            # ── STRIDE ──
            if l_ankle.visibility > 0.4 and r_ankle.visibility > 0.4:
                if max(l_ankle.y, r_ankle.y) > 0.82:
                    if not foot_was_down:
                        stride_count += 1
                        foot_was_down = True
                else:
                    foot_was_down = False
                put_label(frame, f"Strides: {stride_count}", 16, line_h, (255,255,255))

            # ── HIP VELOCITY ──
            if l_hip.visibility > 0.4:
                cx = l_hip.x * proc_w
                if prev_hip_x is not None:
                    vel = abs(cx - prev_hip_x) / tpf
                    velocities.append(vel)
                    put_label(frame, f"Vel: {int(vel)} px/s", 16, line_h*2+4, (0,242,254))
                prev_hip_x = cx

            out.write(frame)

        # ── cleanup ──
        cap.release();  cap  = None
        out.release();  out  = None
        pose.close();   pose = None

        reencode_for_web(raw_out, output_path)
        try:   os.remove(video_path)
        except Exception: pass

        summary = {
            "strides":           stride_count,
            "avg_l_knee":        int(np.mean(l_knee_angs))           if l_knee_angs  else 0,
            "avg_r_knee":        int(np.mean(r_knee_angs))           if r_knee_angs  else 0,
            "avg_release_score": round(float(np.mean(rel_scores)),1) if rel_scores   else 0,
            "avg_velocity":      int(np.mean(velocities))            if velocities   else 0,
        }

        with jobs_lock:
            jobs[job_id] = {
                'status':    'done',
                'video_url': f'/static/{os.path.basename(output_path)}',
                'summary':   summary,
            }

    except Exception as exc:
        print(f"Job {job_id} failed: {exc}")
        try:   os.remove(video_path)
        except Exception: pass
        with jobs_lock:
            jobs[job_id] = {'status': 'error', 'error': str(exc)}
    finally:
        for obj, m in [(pose,'close'),(cap,'release'),(out,'release')]:
            if obj:
                try: getattr(obj, m)()
                except Exception: pass


# ──────────────────────────────────────────────────────
# FLASK ROUTES
# ──────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    job_id      = str(uuid.uuid4())[:8]
    input_path  = os.path.join(UPLOAD_FOLDER, f'input_{job_id}.mp4')
    output_name = f'analyzed_{job_id}.mp4'
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    file.save(input_path)

    for f in os.listdir(OUTPUT_FOLDER):
        if f.startswith('analyzed_') and f != output_name:
            try: os.remove(os.path.join(OUTPUT_FOLDER, f))
            except Exception: pass

    with jobs_lock:
        jobs[job_id] = {'status': 'processing'}

    threading.Thread(
        target=process_bowling_video,
        args=(input_path, output_path, job_id),
        daemon=True,
    ).start()

    return jsonify({'job_id': job_id}), 202

@app.route('/status/<job_id>')
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(job)

@app.route('/static/<filename>')
def serve_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
