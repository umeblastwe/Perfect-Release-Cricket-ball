import cv2
import mediapipe as mp
import os
import numpy as np
import time
import threading  # Background processing ke liye
from flask import Flask, request, render_template, jsonify, make_response, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# --- OPTIMIZATION 1: Global Scope mein MediaPipe load karein ---
mp_pose = mp.solutions.pose
pose_engine = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Processing jobs ka status track karne ke liye global dictionary
JOBS_STATUS = {}

def calculate_joint_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Background task function
def background_processing(job_id, input_path, output_path, epoch_time, output_filename):
    global JOBS_STATUS
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            JOBS_STATUS[job_id] = {"status": "failed", "error": "Cannot open video"}
            return

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        time_per_frame = 1.0 / fps if fps > 0 else 0.0167

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 25.0, (orig_w, orig_h))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 25.0, (orig_w, orig_h))

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
            
            # Global engine use ho rha hai
            results = pose_engine.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=3, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

                landmarks = results.pose_landmarks.landmark
                l_hip, l_knee, l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_HIP], landmarks[mp_pose.PoseLandmark.LEFT_KNEE], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                r_hip, r_knee, r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_HIP], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                l_shoulder, r_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_wrist, r_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                    l_angle = calculate_joint_angle(l_hip, l_knee, l_ankle)
                    l_knee_angles.append(l_angle)

                if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                    r_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
                    r_knee_angles.append(r_angle)

                if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5 and l_shoulder.visibility > 0.5:
                    highest_wrist = l_wrist if l_wrist.y < r_wrist.y else r_wrist
                    corresponding_shoulder = l_shoulder if highest_wrist == l_wrist else r_shoulder
                    release_height_score = (max(l_ankle.y, r_ankle.y) - highest_wrist.y) * 100
                    release_scores.append(release_height_score)
                    dx = highest_wrist.x - corresponding_shoulder.x
                    dy = corresponding_shoulder.y - highest_wrist.y
                    arm_angles.append(np.degrees(np.arctan2(abs(dx), dy)))

                if l_ankle.visibility > 0.5 and r_ankle.visibility > 0.5:
                    if max(l_ankle.y, r_ankle.y) > 0.82:
                        if not foot_was_down:
                            stride_count += 1
                            foot_was_down = True
                    else:
                        foot_was_down = False

                if l_hip.visibility > 0.5:
                    current_hip_x = l_hip.x * w
                    if prev_hip_x is not None:
                        velocities.append(abs(current_hip_x - prev_hip_x) / time_per_frame)
                    prev_hip_x = current_hip_x

            out.write(frame)

        cap.release()
        out.release()

        summary = {
            "strides": stride_count,
            "avg_l_knee": int(np.mean(l_knee_angles)) if l_knee_angles else 0,
            "avg_r_knee": int(np.mean(r_knee_angles)) if r_knee_angles else 0,
            "avg_arm_angle": int(np.mean(arm_angles)) if arm_angles else 0,
            "avg_release_score": round(float(np.mean(release_scores)), 1) if release_scores else 0,
            "avg_velocity": int(np.mean(velocities)) if velocities else 0,
        }

        # Job complete mark karein
        JOBS_STATUS[job_id] = {
            "status": "completed",
            "video_url": f'/static/{output_filename}?v={epoch_time}',
            "summary": summary
        }
    except Exception as e:
        JOBS_STATUS[job_id] = {"status": "failed", "error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

# --- OPTIMIZATION 2: Async Upload Route ---
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    job_id = str(int(time.time()))
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'input_{job_id}.mp4')
    file.save(input_path)

    epoch_time = int(time.time())
    output_filename = f'analyzed_{epoch_id}.mp4' if 'epoch_id' in locals() else f'analyzed_{epoch_time}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    JOBS_STATUS[job_id] = {"status": "processing"}

    # Threading start karein taake request فورا free ho jaye
    threading.Thread(target=background_processing, args=(job_id, input_path, output_path, epoch_time, output_filename)).start()

    return jsonify({'job_id': job_id, 'status': 'processing'})

# --- OPTIMIZATION 3: Polling Endpoint ---
@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    status = JOBS_STATUS.get(job_id, {"status": "not_found"})
    return jsonify(status)

@app.route('/static/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video asset not found'}), 404
    response = make_response(send_file(video_path, mimetype='video/mp4'))
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
