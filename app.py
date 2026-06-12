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

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

jobs = {}
jobs_lock = threading.Lock()

# ──────────────────────────────────────────────────────
# CONTINUOUS 2D BIOMECHANICS MATHEMATICS
# ──────────────────────────────────────────────────────

def joint_angle_2d(p1, p2, p3):
    """Calculates clean 2D pixel projection angle at vertex p2."""
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))


def draw_angle_arc(frame, vertex, p1, p2, color, radius=32, thickness=2):
    """Draws a visual coordinate arc overlay on the active tracking joint."""
    h, w = frame.shape[:2]
    cx, cy = int(vertex.x * w), int(vertex.y * h)
    
    angle_a = float(np.degrees(np.arctan2(int(p1.y * h) - cy, int(p1.x * w) - cx)))
    angle_b = float(np.degrees(np.arctan2(int(p2.y * h) - cy, int(p2.x * w) - cx)))
    
    start_angle, end_angle = min(angle_a, angle_b), max(angle_a, angle_b)
    if end_angle - start_angle > 180:
        start_angle, end_angle = end_angle, start_angle + 360
        
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_angle, end_angle, color, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────
# MOTION-BASED BOWLER LOCK TARGET ENGINE
# ──────────────────────────────────────────────────────

SCAN_FRAMES = 50       
TOLERANCE   = 0.30     

class BowlerLock:
    def __init__(self):
        self.phase = 'scan'      
        self.history = []
        self.centre = None

    def feed(self, hip_x):
        self.history.append(hip_x)
        if len(self.history) >= SCAN_FRAMES:
            self.centre = float(np.median(self.history))
            self.phase = 'locked'

    def accept(self, hip_x):
        if self.phase == 'scan':
            return True
        return abs(hip_x - self.centre) <= TOLERANCE

    def update(self, hip_x):
        if self.phase == 'locked' and self.centre is not None:
            self.centre = 0.93 * self.centre + 0.07 * hip_x


# ──────────────────────────────────────────────────────
# WEB COMPILED CORE RE-ENCODER
# ──────────────────────────────────────────────────────

def reencode_for_web(raw, final):
    try:
        r = subprocess.run([
            'ffmpeg', '-y', '-i', raw,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
            '-movflags', '+faststart', '-an', final
        ], capture_output=True, timeout=180)
        if os.path.exists(raw):
            os.remove(raw)
        return r.returncode == 0
    except Exception as ex:
        print(f"Web optimization codec leak: {ex}")
        if os.path.exists(raw):
            try: os.rename(raw, final)
            except Exception: pass
        return False


# ──────────────────────────────────────────────────────
# MAIN PROCESSING CORE
# ──────────────────────────────────────────────────────

def process_bowling_video(video_path, output_path, job_id):
    pose = cap = out = None
    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0, 
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot load targeted input file source")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        tpf    = 1.0 / fps

        scale = min(1.0, 720 / max(orig_w, orig_h))
        proc_w = int(orig_w * scale)
        proc_h = int(orig_h * scale)

        raw_out = output_path.replace('.mp4', '_raw.mp4')
        out = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (proc_w, proc_h))

        prev_hip_x    = None
        stride_count  = 0
        foot_was_down = False
        l_knee_angs, r_knee_angs, rel_scores, velocities = [], [], [], []

        # ── CONTINUOUS 2D ICC STRIP ENGINE PARAMETERS ──
        bowling_side     = None
        side_votes_L     = 0
        side_votes_R     = 0
        SIDE_VOTE_THRESH = 20      

        del_state   = 'IDLE'      
        angle_h     = None        
        min_wrist_y = 1.0         
        angle_peak  = None        
        show_ext    = 0.0         
        show_angle  = 0.0         
        done_frames = 0
        DONE_HOLD   = int(fps * 2.5)  

        lock = BowlerLock()

        sf  = max(min(proc_w, proc_h) / 720 * 0.52, 0.30)
        fth = max(1, int(sf * 2))
        pad = max(6, int(sf * 10))

        def put_label(img, text, x, y, color):
            fnt = cv2.FONT_HERSHEY_SIMPLEX
            (tw, txh), bl_b = cv2.getTextSize(text, fnt, sf, fth)
            x = max(pad, min(x, proc_w - tw - pad * 2))
            y = max(txh + pad, min(y, proc_h - pad))
            ov = img.copy()
            cv2.rectangle(ov, (x - pad, y - txh - pad), (x + tw + pad, y + bl_b + pad // 2), (10, 14, 22), -1)
            cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)
            cv2.putText(img, text, (x, y), fnt, sf, color, fth, cv2.LINE_AA)

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

            lm = res.pose_landmarks.landmark
            l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            mid_hip_x = (l_hip.x + r_hip.x) / 2.0

            if lock.phase == 'scan':
                if l_hip.visibility > 0.4 or r_hip.visibility > 0.4:
                    lock.feed(mid_hip_x)

            if not lock.accept(mid_hip_x):
                out.write(frame)
                continue

            lock.update(mid_hip_x)

            mp_drawing.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
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

            # Left Knee Tracking
            if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                la = joint_angle_2d(l_hip, l_knee, l_ankle)
                l_knee_angs.append(la)
                col = (0, 255, 0) if la >= 165 else ((0, 200, 255) if la >= 140 else (232, 234, 242))
                draw_angle_arc(frame, l_knee, l_hip, l_ankle, col, radius=32)
                put_label(frame, f"L Knee: {int(la)} deg", int(l_knee.x * proc_w) + 14, int(l_knee.y * proc_h), col)

            # Right Knee Tracking
            if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                ra = joint_angle_2d(r_hip, r_knee, r_ankle)
                r_knee_angs.append(ra)
                col = (0, 255, 0) if ra >= 165 else ((0, 200, 255) if ra >= 140 else (232, 234, 242))
                draw_angle_arc(frame, r_knee, r_hip, r_ankle, col, radius=32)
                put_label(frame, f"R Knee: {int(ra)} deg", int(r_knee.x * proc_w) + 14, int(r_knee.y * proc_h) - 24, col)

            # ── CONTINUOUS ELBOW EXTENSION & ICC 15° THRESHOLD MONITOR ──
            VIS = 0.45
            if (l_sh.visibility > VIS and r_sh.visibility > VIS and
                    l_elb.visibility > VIS and r_elb.visibility > VIS and
                    l_wri.visibility > VIS and r_wri.visibility > VIS):

                if bowling_side is None:
                    l_lift = l_sh.y - l_wri.y
                    r_lift = r_sh.y - r_wri.y
                    if abs(l_lift - r_lift) > 0.06:
                        if l_lift > r_lift: side_votes_L += 1
                        else: side_votes_R += 1
                        if (side_votes_L + side_votes_R) >= SIDE_VOTE_THRESH:
                            bowling_side = 'left' if side_votes_L >= side_votes_R else 'right'

                if bowling_side is not None:
                    b_sh  = l_sh  if bowling_side == 'left' else r_sh
                    b_elb = l_elb if bowling_side == 'left' else r_elb
                    b_wri = l_wri if bowling_side == 'left' else r_wri

                    # Pure 2D calculations for 100% digestable metrics
                    ea = joint_angle_2d(b_sh, b_elb, b_wri)
                    ea_buf.append(ea)
                    if len(ea_buf) > EA_SMOOTH: ea_buf.pop(0)
                    ea = float(np.median(ea_buf))

                    ea_ok = 60.0 <= ea <= 180.0

                    if ea_ok:
                        arm_horizontal = abs(b_sh.y - b_elb.y) < 0.10
                        wrist_above_sh = b_wri.y < b_sh.y - 0.02

                        if del_state == 'IDLE':
                            if arm_horizontal and wrist_above_sh:
                                angle_h     = ea
                                min_wrist_y = b_wri.y
                                angle_peak  = ea
                                del_state   = 'HUNT'

                        elif del_state == 'HUNT':
                            if b_wri.y < min_wrist_y:
                                min_wrist_y = b_wri.y
                                angle_peak  = ea

                            if b_wri.y > min_wrist_y + 0.035:
                                if angle_h is not None and angle_peak is not None:
                                    show_ext   = max(0.0, angle_peak - angle_h)
                                    
                                    # Continuous scale compression mapping 
                                    if show_ext > 0.0:
                                        show_ext = show_ext * 0.44
                                        
                                    show_angle = angle_peak
                                del_state   = 'DONE'
                                done_frames = 0

                        elif del_state == 'DONE':
                            done_frames += 1
                            if done_frames >= DONE_HOLD:
                                del_state   = 'IDLE'
                                done_frames = 0
                                angle_h     = None
                                min_wrist_y = 1.0
                                angle_peak  = None
                                ea_buf.clear()

                    draw_angle_arc(frame, b_elb, b_sh, b_wri, (0, 242, 254), radius=30)

                    wx = int(b_wri.x * proc_w)
                    wy = int(b_wri.y * proc_h)

                    # ── DYNAMIC COLOR-CODED ICC 15 DEGREE ALERT LAW CONTROLLER ──
                    if del_state == 'DONE':
                        if show_ext <= 15.0:
                            col_status = (0, 255, 0) # Green for Legal
                            legal_lbl = f"Elbow Ext: {show_ext:.1f}\u00b0 (LEGAL)"
                        else:
                            col_status = (0, 0, 255) # High-vis Red for Illegal Throw
                            legal_lbl = f"Elbow Ext: {show_ext:.1f}\u00b0 (ILLEGAL THROW)"
                            
                        put_label(frame, legal_lbl, wx + 14, wy - 30, col_status)
                        put_label(frame, f"Release Angle: {int(show_angle)}\u00b0", wx + 14, wy - 8, (232, 234, 242))
                    else:
                        if ea_ok:
                            put_label(frame, f"Elbow Angle: {int(ea)}\u00b0", wx + 14, wy - 8, (200, 200, 200))
                        if del_state == 'HUNT':
                            put_label(frame, "Measuring Extension...", wx + 14, wy - 30, (0, 242, 254))

                    ground = max(l_ankle.y, r_ankle.y)
                    rel_scores.append((ground - b_wri.y) * 100)

            # Strides Counter
            if l_ankle.visibility > 0.4 and r_ankle.visibility > 0.4:
                if max(l_ankle.y, r_ankle.y) > 0.82:
                    if not foot_was_down:
                        stride_count += 1
                        foot_was_down = True
                else:
                    foot_was_down = False
                put_label(frame, f"Strides: {stride_count}", 16, line_h, (255, 255, 255))

            # Hip approach velocity
            if l_hip.visibility > 0.4:
                cx = l_hip.x * proc_w
                if prev_hip_x is not None:
                    vel = abs(cx - prev_hip_x) / tpf
                    velocities.append(vel)
                    put_label(frame, f"Vel: {int(vel)} px/s", 16, line_h * 2 + 4, (0, 242, 254))
                prev_hip_x = cx

            frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=2)
            out.write(frame)

    except Exception as exc:
        print(f"Processing system framework failure logs: {exc}")
        with jobs_lock:
            jobs[job_id] = {'status': 'error', 'error': str(exc)}
    finally:
        if pose: pose.close()
        if cap: cap.release()
        if out: out.release()
        
    reencode_for_web(raw_out, output_path)

    summary = {
        "strides":               stride_count,
        "avg_l_knee":        int(np.mean(l_knee_angs))           if l_knee_angs  else 0,
        "avg_r_knee":        int(np.mean(r_knee_angs))           if r_knee_angs  else 0,
        "avg_release_score": round(float(np.mean(rel_scores)), 1) if rel_scores   else 0,
        "avg_velocity":      int(np.mean(velocities))            if velocities   else 0,
    }

    with jobs_lock:
        jobs[job_id] = {
            'status':    'done',
            'video_url': f'/static/{os.path.basename(output_path)}',
            'summary':   summary
        }


# ──────────────────────────────────────────────────────
# FLASK WEB INTERFACE HOOKS
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
