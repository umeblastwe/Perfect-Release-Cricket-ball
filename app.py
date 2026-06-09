import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
import time
import threading
import uuid
from flask import Flask, request, render_template, jsonify, send_file
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
    """Generic angle at vertex p2 between p1-p2-p3."""
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def reencode_for_web(raw_path, final_path):
    """ffmpeg re-encode with faststart for browser playback."""
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', raw_path,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
            '-movflags', '+faststart',
            '-an',  # No audio to preserve Render container RAM
            final_path
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
    """Runs in background thread. Wipes MediaPipe immediately after use to protect RAM."""
    pose = None
    cap = None
    out = None
    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # mandatory for 512MB Render tier
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
        out = cv2.VideoWriter(raw_output, fourcc, fps, (proc_w, proc_h))
        if not out.isOpened():
            raise RuntimeError("Could not create video writer")

        prev_hip_x  = None
        stride_count = 0
        foot_was_down = False
        l_knee_angles, r_knee_angles = [], []
        release_scores, velocities = [], []

        # Release Snapshot frame tracking variables
        wrist_peak_y = None
        release_snapshot_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if (proc_w, proc_h) != (orig_w, orig_h):
                frame = cv2.resize(frame, (proc_w, proc_h))

            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            scale_font = max(min(w, h) / 720 * 0.5, 0.28)
            thickness = max(1, int(scale_font * 2))
            pad       = int(scale_font * 12)

            def put_label(img, text, x, y, color):
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), bl = cv2.getTextSize(text, font, scale_font, thickness)
                x = max(pad, min(x, w - tw - pad * 2))
                y = max(th + pad, min(y, h - pad))
                overlay = img.copy()
                cv2.rectangle(overlay, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad//2), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
                cv2.putText(img, text, (x, y), font, scale_font, color, thickness, cv2.LINE_AA)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,242,254), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=1)
                )

                lm = results.pose_landmarks.landmark
                l_hip   = lm[mp_pose.PoseLandmark.LEFT_HIP]
                l_knee  = lm[mp_pose.PoseLandmark.LEFT_KNEE]
                l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                r_hip   = lm[mp_pose.PoseLandmark.RIGHT_HIP]
                r_knee  = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
                r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

                if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                    l_angle = calculate_joint_angle(l_hip, l_knee, l_ankle)
                    l_knee_angles.append(l_angle)
                    col = (0,255,0) if l_angle > 165 else (200,200,200)
                    put_label(frame, f"L {int(l_angle)}\u00b0", int(l_knee.x*w)+12, int(l_knee.y*h), col)

                if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                    r_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
                    r_knee_angles.append(r_angle)
                    col = (0,255,0) if r_angle > 165 else (200,200,200)
                    put_label(frame, f"R {int(r_angle)}\u00b0", int(r_knee.x*w)+12, int(r_knee.y*h)-20, col)

                # Track Release Height Point & Snapshot Trigger
                if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5:
                    highest_wrist = l_wrist if l_wrist.y < r_wrist.y else r_wrist
                    
                    # Capture snapshot when wrist reaches its highest physical elevation (minimal y profile)
                    if wrist_peak_y is None or highest_wrist.y < wrist_peak_y:
                        wrist_peak_y = highest_wrist.y
                        
                        wx = int(highest_wrist.x * w)
                        wy = int(highest_wrist.y * h)
                        
                        # Generate labeled base canvas slice
                        release_snapshot_frame = frame.copy()
                        put_label(release_snapshot_frame, "RELEASE POINT DETECTED", wx + 14, wy - 15, (0, 242, 254))

                    ground_ref = max(l_ankle.y, r_ankle.y)
                    rel_score  = (ground_ref - highest_wrist.y) * 100
                    release_scores.append(rel_score)

                line_h = int(scale_font * 36)
                if l_ankle.visibility > 0.5 and r_ankle.visibility > 0.5:
                    if max(l_ankle.y, r_ankle.y) > 0.82:
                        if not foot_was_down:
                            stride_count += 1
                            foot_was_down = True
                    else:
                        foot_was_down = False
                    put_label(frame, f"Strides: {stride_count}", 16, line_h, (255,255,255))

                if l_hip.visibility > 0.5:
                    curr_x = l_hip.x * w
                    if prev_hip_x is not None:
                        vel = abs(curr_x - prev_hip_x) / time_per_frame
                        velocities.append(vel)
                        put_label(frame, f"Vel: {int(vel)} px/s", 16, line_h*2+4, (0,242,254))
                    prev_hip_x = curr_x

            out.write(frame)

        cap.release()
        cap = None
        out.release()
        out = None
        pose.close()
        pose = None

        # Process snapshot export logic
        snapshot_filename = None
        if release_snapshot_frame is not None:
            snapshot_filename = f"snapshot_{job_id}.jpg"
            snapshot_path = os.path.join(OUTPUT_FOLDER, snapshot_filename)
            cv2.imwrite(snapshot_path, release_snapshot_frame)

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
                'status': 'done',
                'video_url': f'/static/{os.path.basename(output_path)}',
                'snapshot_url': f'/static/{snapshot_filename}' if snapshot_filename else None,
                'summary': summary
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

    job_id = str(uuid.uuid4())[:8]
    input_path  = os.path.join(UPLOAD_FOLDER, f'input_{job_id}.mp4')
    output_name = f'analyzed_{job_id}.mp4'
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    file.save(input_path)

    # Clean legacy run objects
    for f in os.listdir(OUTPUT_FOLDER):
        if (f.startswith('analyzed_') or f.startswith('snapshot_')) and f != output_name:
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


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
