import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
import time
import threading
from flask import Flask, request, render_template, jsonify, send_file, make_response
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

# Global safe pipeline pooling engines
mp_pose = mp.solutions.pose
pose_engine = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

JOBS_STATUS = {}

def calculate_joint_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def async_video_processing(job_id, input_path, output_path, output_filename, epoch_time):
    global JOBS_STATUS
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            JOBS_STATUS[job_id] = {"status": "failed", "error": "Invalid video codec mapping stream properties."}
            return

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 25.0
        time_per_frame = 1.0 / fps

        # FIXED: Directly use forced universal mp4v container layer to bypass missing system encoders
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

        prev_hip_x = None
        stride_count = 0
        foot_was_down = False
        l_knee_angles, r_knee_angles, arm_angles, release_scores, velocities = [], [], [], [], []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_engine.process(rgb_frame)

            scale = max(min(w, h) / 1080 * 0.55, 0.28)
            thickness = max(1, int(scale * 2.2))
            pad = int(scale * 14)

            def put_label(img, text, x, y, color, bg_alpha=0.55):
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
                x = max(pad, min(x, w - tw - pad * 2))
                y = max(th + pad, min(y, h - pad))
                overlay = img.copy()
                cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad // 2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
                cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                )

                landmarks = results.pose_landmarks.landmark
                l_hip, l_knee, l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.LEFT_KNEE], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                r_hip, r_knee, r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                l_shoulder, r_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_wrist, r_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                    l_angle = calculate_joint_angle(l_hip, l_knee, l_ankle)
                    l_knee_angles.append(l_angle)
                    l_color = (34, 197, 94) if l_angle > 165 else (255, 255, 255)
                    put_label(frame, f"L {int(l_angle)}\u00b0", int(l_knee.x * w) + 12, int(l_knee.y * h), l_color)

                if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                    r_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
                    r_knee_angles.append(r_angle)
                    r_color = (34, 197, 94) if r_angle > 165 else (255, 255, 255)
                    put_label(frame, f"R {int(r_angle)}\u00b0", int(r_knee.x * w) + 12, int(r_knee.y * h) - 20, r_color)

                if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5 and l_shoulder.visibility > 0.5:
                    highest_wrist = l_wrist if l_wrist.y < r_wrist.y else r_wrist
                    corresponding_shoulder = l_shoulder if highest_wrist == l_wrist else r_shoulder
                    release_height_score = (max(l_ankle.y, r_ankle.y) - highest_wrist.y) * 100
                    release_scores.append(release_height_score)
                    dx = highest_wrist.x - corresponding_shoulder.x
                    dy = corresponding_shoulder.y - highest_wrist.y
                    arm_angle_deg = np.degrees(np.arctan2(abs(dx), dy))
                    arm_angles.append(arm_angle_deg)

                    put_label(frame, f"REL {int(release_height_score)}", int(highest_wrist.x * w) + 14, int(highest_wrist.y * h) - 30, (0, 242, 254))
                    put_label(frame, f"ARM {int(arm_angle_deg)}\u00b0", int(highest_wrist.x * w) + 14, int(highest_wrist.y * h) - 6, (245, 230, 66))

                line_h = int(scale * 38)
                if l_ankle.visibility > 0.5 and r_ankle.visibility > 0.5:
                    if max(l_ankle.y, r_ankle.y) > 0.82:
                        if not foot_was_down:
                            stride_count += 1
                            foot_was_down = True
                    else:
                        foot_was_down = False
                put_label(frame, f"Strides: {stride_count}", 16, line_h, (255, 255, 255))

                if l_hip.visibility > 0.5:
                    current_hip_x = l_hip.x * w
                    if prev_hip_x is not None:
                        velocities.append(abs(current_hip_x - prev_hip_x) / time_per_frame)
                    put_label(frame, f"Vel: {int(np.mean(velocities)) if velocities else 0} px/s", 16, line_h * 2 + 4, (0, 242, 254))
                    prev_hip_x = current_hip_x

            out.write(frame)

        cap.release()
        out.write(frame) if 'frame' in locals() else None
        out.release()

        summary = {
            "strides": stride_count,
            "avg_l_knee": int(np.mean(l_knee_angles)) if l_knee_angles else 0,
            "avg_r_knee": int(np.mean(r_knee_angles)) if r_knee_angles else 0,
            "avg_arm_angle": int(np.mean(arm_angles)) if arm_angles else 0,
            "avg_release_score": round(float(np.mean(release_scores)), 1) if release_scores else 0,
            "avg_velocity": int(np.mean(velocities)) if velocities else 0,
        }

        JOBS_STATUS[job_id] = {
            "status": "completed",
            "video_url": f'/static/{output_filename}',
            "summary": summary
        }
    except Exception as e:
        JOBS_STATUS[job_id] = {"status": "failed", "error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file payload located.'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Null filename identifier.'}), 400

    job_id = str(int(time.time()))
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'input_{job_id}.mp4')
    file.save(input_path)

    epoch_time = int(time.time())
    output_filename = f'analyzed_{epoch_time}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    # Temporary garbage management execution clear
    for f in os.listdir(app.config['OUTPUT_FOLDER']):
        if f.startswith('analyzed_') and f != output_filename:
            try: os.remove(os.path.join(app.config['OUTPUT_FOLDER'], f))
            except Exception: pass

    JOBS_STATUS[job_id] = {"status": "processing"}
    
    # Executing multi-threaded architecture worker isolation state
    threading.Thread(target=async_video_processing, args=(job_id, input_path, output_path, output_filename, epoch_time)).start()
    
    return jsonify({'job_id': job_id, 'status': 'processing'})

@app.route('/status/<job_id>', methods=['GET'])
def check_job_status(job_id):
    return jsonify(JOBS_STATUS.get(job_id, {"status": "not_found"}))

@app.route('/static/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Resource path not found.'}), 404
    return send_file(video_path, mimetype='video/mp4', conditional=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
