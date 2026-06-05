import cv2
import mediapipe as mp
import os
import numpy as np
import time
import threading
import uuid
from flask import Flask, request, render_template, jsonify, make_response, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Global job status store
JOBS_STATUS = {}
JOBS_LOCK   = threading.Lock()


# ─────────────────────────────────────────────
# Helper: create a fresh per-thread MediaPipe pose
# (MediaPipe is NOT thread-safe — never share one instance)
# ─────────────────────────────────────────────
def make_pose():
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def calculate_joint_angle(p1, p2, p3):
    a  = np.array([p1.x, p1.y])
    b  = np.array([p2.x, p2.y])
    c  = np.array([p3.x, p3.y])
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos_a = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def _try_writer(path, fourcc_str, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    return writer if writer.isOpened() else None


# ─────────────────────────────────────────────
# Background processing — one fresh pose per call
# ─────────────────────────────────────────────
def background_processing(job_id, input_path, output_path, output_filename):
    mp_pose   = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        time_per_frame = 1.0 / fps
        size   = (orig_w, orig_h)

        # Try codecs in order of mobile compatibility
        out = (_try_writer(output_path, 'avc1', fps, size) or
               _try_writer(output_path, 'H264', fps, size) or
               _try_writer(output_path, 'mp4v', fps, size))

        if out is None:
            raise RuntimeError("No working video codec found on this system")

        prev_hip_x   = None
        stride_count = 0
        foot_was_down = False
        l_knee_angles, r_knee_angles, arm_angles, release_scores, velocities = [], [], [], [], []

        # Create a fresh pose instance for this thread
        with make_pose() as pose_engine:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose_engine.process(rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=3, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                    )

                    lm = results.pose_landmarks.landmark
                    l_hip,  l_knee,  l_ankle  = lm[mp_pose.PoseLandmark.LEFT_HIP],  lm[mp_pose.PoseLandmark.LEFT_KNEE],  lm[mp_pose.PoseLandmark.LEFT_ANKLE]
                    r_hip,  r_knee,  r_ankle  = lm[mp_pose.PoseLandmark.RIGHT_HIP], lm[mp_pose.PoseLandmark.RIGHT_KNEE], lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    l_shoulder, r_shoulder    = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    l_wrist,    r_wrist       = lm[mp_pose.PoseLandmark.LEFT_WRIST],    lm[mp_pose.PoseLandmark.RIGHT_WRIST]

                    # Left knee angle
                    if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                        l_angle = calculate_joint_angle(l_hip, l_knee, l_ankle)
                        l_knee_angles.append(l_angle)
                        l_col = (0, 255, 0) if l_angle > 165 else (200, 200, 200)
                        cv2.putText(frame, f"L Knee: {int(l_angle)} deg",
                                    (int(l_knee.x * w) + 20, int(l_knee.y * h)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, l_col, 3, cv2.LINE_AA)

                    # Right knee angle
                    if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                        r_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
                        r_knee_angles.append(r_angle)
                        r_col = (0, 255, 0) if r_angle > 165 else (200, 200, 200)
                        cv2.putText(frame, f"R Knee: {int(r_angle)} deg",
                                    (int(r_knee.x * w) + 20, int(r_knee.y * h) - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, r_col, 3, cv2.LINE_AA)

                    # Wrist / arm analysis
                    if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5 and l_shoulder.visibility > 0.5:
                        highest_wrist       = l_wrist if l_wrist.y < r_wrist.y else r_wrist
                        corr_shoulder       = l_shoulder if highest_wrist is l_wrist else r_shoulder
                        ground_ref          = max(l_ankle.y, r_ankle.y)
                        release_score       = (ground_ref - highest_wrist.y) * 100
                        release_scores.append(release_score)
                        dx = highest_wrist.x - corr_shoulder.x
                        dy = corr_shoulder.y  - highest_wrist.y
                        arm_angle = np.degrees(np.arctan2(abs(dx), dy))
                        arm_angles.append(arm_angle)

                        px = int(highest_wrist.x * w)
                        py = int(highest_wrist.y * h)
                        cv2.putText(frame, f"HAND REL: {int(release_score)} pts",
                                    (px + 25, py - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 242, 254), 3, cv2.LINE_AA)
                        cv2.putText(frame, f"ARM ANGLE: {int(arm_angle)} deg",
                                    (px + 25, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 242, 0), 3, cv2.LINE_AA)

                    # Stride detection
                    if l_ankle.visibility > 0.5 and r_ankle.visibility > 0.5:
                        if max(l_ankle.y, r_ankle.y) > 0.82:
                            if not foot_was_down:
                                stride_count += 1
                                foot_was_down = True
                        else:
                            foot_was_down = False
                        cv2.putText(frame, f"Strides: {stride_count}", (40, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

                    # Hip velocity
                    if l_hip.visibility > 0.5:
                        curr_hip_x = l_hip.x * w
                        if prev_hip_x is not None:
                            vel = abs(curr_hip_x - prev_hip_x) / time_per_frame
                            velocities.append(vel)
                            cv2.putText(frame, f"Velocity: {int(vel)} px/s", (40, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 242, 254), 3, cv2.LINE_AA)
                        prev_hip_x = curr_hip_x

                out.write(frame)

        cap.release()
        out.release()

        summary = {
            "strides":           stride_count,
            "avg_l_knee":        int(np.mean(l_knee_angles))    if l_knee_angles    else 0,
            "avg_r_knee":        int(np.mean(r_knee_angles))    if r_knee_angles    else 0,
            "avg_arm_angle":     int(np.mean(arm_angles))       if arm_angles       else 0,
            "avg_release_score": round(float(np.mean(release_scores)), 1) if release_scores else 0,
            "avg_velocity":      int(np.mean(velocities))       if velocities       else 0,
        }

        with JOBS_LOCK:
            JOBS_STATUS[job_id] = {
                "status":    "completed",
                "video_url": f"/static/{output_filename}",
                "summary":   summary,
            }

    except Exception as e:
        with JOBS_LOCK:
            JOBS_STATUS[job_id] = {"status": "failed", "error": str(e)}
    finally:
        # Clean up the uploaded source file to save disk space on Render
        try:
            os.remove(input_path)
        except OSError:
            pass


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Use a UUID so multiple users never collide
    job_id          = str(uuid.uuid4())
    input_path      = os.path.join(app.config['UPLOAD_FOLDER'], f'input_{job_id}.mp4')
    output_filename = f'analyzed_{job_id}.mp4'
    output_path     = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    file.save(input_path)

    with JOBS_LOCK:
        JOBS_STATUS[job_id] = {"status": "processing"}

    threading.Thread(
        target=background_processing,
        args=(job_id, input_path, output_path, output_filename),
        daemon=True,
    ).start()

    return jsonify({'job_id': job_id, 'status': 'processing'})


@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    with JOBS_LOCK:
        status = JOBS_STATUS.get(job_id, {"status": "not_found"})
    return jsonify(status)


@app.route('/static/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404
    response = make_response(send_file(video_path, mimetype='video/mp4'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
