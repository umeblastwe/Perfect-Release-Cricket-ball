import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
import time
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


def calculate_joint_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def calculate_3d_elbow_angle(shoulder, elbow, wrist):
    s = np.array([shoulder.x, shoulder.y, shoulder.z])
    e = np.array([elbow.x, elbow.y, elbow.z])
    w = np.array([wrist.x, wrist.y, wrist.z])
    v_se = s - e
    v_ew = w - e
    dot_product = np.dot(v_se, v_ew)
    cosine_angle = dot_product / (np.linalg.norm(v_se) * np.linalg.norm(v_ew) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def draw_angle_arc(frame, vertex, p1, p2, angle_deg, color, radius=32, thickness=2):
    h, w = frame.shape[:2]
    cx = int(vertex.x * w)
    cy = int(vertex.y * h)
    ax = int(p1.x * w) - cx
    ay = int(p1.y * h) - cy
    bx = int(p2.x * w) - cx
    by = int(p2.y * h) - cy
    angle_a = np.degrees(np.arctan2(ay, ax))
    angle_b = np.degrees(np.arctan2(by, bx))
    start_angle = min(angle_a, angle_b)
    end_angle   = max(angle_a, angle_b)
    if end_angle - start_angle > 180:
        start_angle, end_angle = end_angle, start_angle + 360
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_angle, end_angle, color, thickness, cv2.LINE_AA)


def reencode_for_web(raw_path, final_path):
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', raw_path,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
            '-movflags', '+faststart', '-an', final_path
        ], capture_output=True, timeout=180)
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return result.returncode == 0
    except Exception as e:
        print(f"ffmpeg error: {e}")
        if os.path.exists(raw_path):
            try: os.rename(raw_path, final_path)
            except Exception: pass
        return False


def process_bowling_video(video_path, output_path, job_id):
    pose = None
    cap  = None
    out  = None
    try:
        mp_pose    = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.4, # Adjusted for high-speed dynamic tracking locks
            min_tracking_confidence=0.4,
            model_complexity=0
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        time_per_frame = 1.0 / fps

        max_dim = 720
        if max(orig_w, orig_h) > max_dim:
            scale_dims = max_dim / max(orig_w, orig_h)
            proc_w = int(orig_w * scale_dims)
            proc_h = int(orig_h * scale_dims)
        else:
            proc_w, proc_h = orig_w, orig_h

        raw_output = output_path.replace('.mp4', '_raw.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(raw_output, fourcc, fps, (proc_w, proc_h))

        prev_hip_x    = None
        stride_count  = 0
        foot_was_down = False
        l_knee_angles, r_knee_angles = [], []
        release_scores, velocities   = [], []

        theta_horizontal         = None
        arm_reached_horizontal   = False
        bowling_side             = None

        # ── BACKGROUND MOTION FILTER VARIABLE SETUP ──
        fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=24, detectShadows=False)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if (proc_w, proc_h) != (orig_w, orig_h):
                frame = cv2.resize(frame, (proc_w, proc_h))

            h, w = frame.shape[:2]
            
            # ── MOTION TARGET DETECTION CONTROLLER ──
            # Apply optical motion density mapping to extract the bounding box coordinates of the RUNNING bowler
            fgmask = fgbg.apply(frame)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bowler_bbox = None
            max_area = 0
            
            # Filter out tiny pixel adjustments and locate the primary running figure
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1200: # Threshold configuration to avoid umpire hand signals
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    # Bowler runs vertically or diagonally down the pitch profile tracks
                    if area > max_area:
                        max_area = area
                        bowler_bbox = (x, y, w_box, h_box)

            # Define isolated Region of Interest slice to lock processing pipeline on bowler canvas
            analysis_target_frame = frame.copy()
            if bowler_bbox is not None:
                x, y, w_box, h_box = bowler_bbox
                # Add directional padding headroom safety markers
                pad_x = int(w_box * 0.25)
                pad_y = int(h_box * 0.25)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + w_box + pad_x)
                y2 = min(h, y + h_box + pad_y)
                
                # Blackout spatial background environments containing static umpires/bystanders
                mask = np.zeros_like(frame)
                mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                rgb_frame = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb_frame)

            scale_font = max(min(w, h) / 720 * 0.5, 0.28)
            thickness  = max(1, int(scale_font * 2))
            pad        = int(scale_font * 12)

            def put_label(img, text, x, y, color):
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), bl = cv2.getTextSize(text, font, scale_font, thickness)
                x = max(pad, min(x, w - tw - pad * 2))
                y = max(th + pad, min(y, h - pad))
                overlay = img.copy()
                cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad // 2), (12, 16, 24), -1)
                cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
                cv2.putText(img, text, (x, y), font, scale_font, color, thickness, cv2.LINE_AA)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                l_hip      = lm[mp_pose.PoseLandmark.LEFT_HIP]
                l_knee     = lm[mp_pose.PoseLandmark.LEFT_KNEE]
                l_ankle    = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                r_hip      = lm[mp_pose.PoseLandmark.RIGHT_HIP]
                r_knee     = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
                r_ankle    = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_wrist    = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                r_wrist    = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                l_elbow    = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
                r_elbow    = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]

                # ── BIOMECHANICAL BOUNDARY SANITY CONTROLLER ──
                # If the tracker targets an entity too far from the computed dynamic motion epicenter (e.g., Umpire), abort loop processing
                if bowler_bbox is not None:
                    hip_pixel_x = int(l_hip.x * w)
                    if hip_pixel_x < x1 or hip_pixel_x > x2:
                        out.write(frame)
                        continue

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                )

                # Left Knee
                if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                    l_angle = calculate_joint_angle(l_hip, l_knee, l_ankle)
                    l_knee_angles.append(l_angle)
                    arc_col = (0, 255, 0) if l_angle >= 165 else ((0, 200, 255) if l_angle >= 140 else (232, 234, 242))
                    draw_angle_arc(frame, l_knee, l_hip, l_ankle, l_angle, arc_col, radius=32)
                    put_label(frame, f"L Knee: {int(l_angle)}\u00b0", int(l_knee.x*w)+12, int(l_knee.y*h), arc_col)

                # Right Knee
                if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                    r_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
                    r_knee_angles.append(r_angle)
                    arc_col_r = (0, 255, 0) if r_angle >= 165 else ((0, 200, 255) if r_angle >= 140 else (232, 234, 242))
                    draw_angle_arc(frame, r_knee, r_hip, r_ankle, r_angle, arc_col_r, radius=32)
                    put_label(frame, f"R Knee: {int(r_angle)}\u00b0", int(r_knee.x*w)+12, int(r_knee.y*h)-22, arc_col_r)

                # ICC 3D Elbow Extension 
                if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5:
                    if bowling_side is None:
                        bowling_side = 'left' if l_wrist.y < r_wrist.y else 'right'

                    b_shoulder = l_shoulder if bowling_side == 'left' else r_shoulder
                    b_elbow    = l_elbow    if bowling_side == 'left' else r_elbow
                    b_wrist    = l_wrist    if bowling_side == 'left' else r_wrist

                    current_3d_angle = calculate_3d_elbow_angle(b_shoulder, b_elbow, b_wrist)

                    arm_is_horizontal = abs(b_shoulder.y - b_elbow.y) < 0.15
                    if arm_is_horizontal and not arm_reached_horizontal:
                        theta_horizontal       = current_3d_angle
                        arm_reached_horizontal = True

                    live_extension = 0.0
                    if arm_reached_horizontal and theta_horizontal is not None:
                        live_extension = max(0.0, current_3d_angle - theta_horizontal)
                        if live_extension > 0.0:
                            live_extension = live_extension * 0.44

                    if b_wrist.y > b_shoulder.y + 0.20:
                        arm_reached_horizontal = False
                        theta_horizontal       = None

                    wx = int(b_wrist.x * w)
                    wy = int(b_wrist.y * h)

                    draw_angle_arc(frame, b_elbow, b_shoulder, b_wrist, current_3d_angle, (0, 242, 254), radius=28)
                    put_label(frame, f"Elbow Ext: {int(live_extension)}\u00b0", wx + 14, wy - 30, (0, 242, 254))
                    put_label(frame, f"Elbow Angle: {int(current_3d_angle)}\u00b0", wx + 14, wy - 6,  (200, 200, 200))

                    ground_ref = max(l_ankle.y, r_ankle.y)
                    rel_score  = (ground_ref - b_wrist.y) * 100
                    release_scores.append(rel_score)

                # Stride Counter
                line_h = int(scale_font * 36)
                if l_ankle.visibility > 0.5 and r_ankle.visibility > 0.5:
                    if max(l_ankle.y, r_ankle.y) > 0.82:
                        if not foot_was_down:
                            stride_count  += 1
                            foot_was_down  = True
                    else:
                        foot_was_down = False
                    put_label(frame, f"Strides: {stride_count}", 16, line_h, (255, 255, 255))

                # Hip Velocity
                if l_hip.visibility > 0.5:
                    curr_x = l_hip.x * w
                    if prev_hip_x is not None:
                        vel = abs(curr_x - prev_hip_x) / time_per_frame
                        velocities.append(vel)
                        put_label(frame, f"Vel: {int(vel)} px/s", 16, line_h * 2 + 4, (0, 242, 254))
                    prev_hip_x = curr_x

            out.write(frame)

        cap.release(); cap = None
        out.release(); out = None
        pose.close();  pose = None

        reencode_for_web(raw_output, output_path)

        if os.path.exists(video_path):
            try: os.remove(video_path)
            except Exception: pass

        summary = {
            "strides":               stride_count,
            "avg_l_knee":        int(np.mean(l_knee_angles))    if l_knee_angles   else 0,
            "avg_r_knee":        int(np.mean(r_knee_angles))    if r_knee_angles   else 0,
            "avg_release_score": round(float(np.mean(release_scores)), 1) if release_scores else 0,
            "avg_velocity":      int(np.mean(velocities))        if velocities      else 0,
        }

        with jobs_lock:
            jobs[job_id] = {
                'status':    'done',
                'video_url': f'/static/{os.path.basename(output_path)}',
                'summary':   summary
            }

    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except Exception: pass
        with jobs_lock:
            jobs[job_id] = {'status': 'error', 'error': str(e)}
    finally:
        if pose:
            try: pose.close()
            except Exception: pass
        if cap:
            try: cap.release()
            except Exception: pass
        if out:
            try: out.release()
            except Exception: pass


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

    t = threading.Thread(
        target=process_bowling_video,
        args=(input_path, output_path, job_id),
        daemon=True
    )
    t.start()
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
