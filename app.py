import cv2
import mediapipe as mp
import os
import numpy as np
import time
import gc  # Garbage collector for clearing RAM
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
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Cap at 50MB to save RAM


def calculate_joint_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def process_bowling_video(video_path, output_path):
    # Core RAM Optimization: Model complexity ko 1 se hata kar 0 (Lightweight) kar diya
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # 0 = Fastest/Lowest RAM, 1 = Balanced, 2 = Heavy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, {}

    # Read dimensions
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    time_per_frame = 1.0 / fps

    # RAM OPTIMIZATION: Max processing width 640px rakhein taake server crash na ho
    target_w = 640
    scale = target_w / float(orig_w) if orig_w > target_w else 1.0
    process_w = int(orig_w * scale)
    process_h = int(orig_h * scale)

    # Use standard mp4v but with downsampled frame dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (process_w, process_h))

    if not out.isOpened():
        cap.release()
        return False, {}

    prev_hip_x = None
    stride_count = 0
    foot_was_down = False

    l_knee_angles = []
    r_knee_angles = []
    arm_angles = []
    release_scores = []
    velocities = []

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # RAM OPTIMIZATION: Frame skipping (Har alternate frame process karein to save 50% memory)
        # Agar processing mazeed optimize karni ho to 'if frame_count % 2 != 0: continue' use kar sakte hain

        # Resize frame to save severe RAM spikes
        if scale < 1.0:
            frame = cv2.resize(frame, (process_w, process_h), interpolation=cv2.INTER_AREA)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
            )

            landmarks = results.pose_landmarks.landmark
            l_hip    = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            l_knee   = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            l_ankle  = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            r_hip    = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            r_knee   = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            r_ankle  = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_wrist  = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wrist  = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Left Knee
            if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                l_angle = calculate_joint_angle(l_hip, l_knee, l_ankle)
                l_knee_angles.append(l_angle)
                cv2.putText(frame, f"L Knee: {int(l_angle)} deg", (20, process_h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Right Knee
            if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                r_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
                r_knee_angles.append(r_angle)
                cv2.putText(frame, f"R Knee: {int(r_angle)} deg", (20, process_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Arm Release
            if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5 and l_shoulder.visibility > 0.5:
                highest_wrist = l_wrist if l_wrist.y < r_wrist.y else r_wrist
                corresponding_shoulder = l_shoulder if highest_wrist == l_wrist else r_shoulder

                ground_reference = max(l_ankle.y, r_ankle.y)
                release_height_score = (ground_reference - highest_wrist.y) * 100
                release_scores.append(release_height_score)

                dx = highest_wrist.x - corresponding_shoulder.x
                dy = corresponding_shoulder.y - highest_wrist.y
                arm_angle_deg = np.degrees(np.arctan2(abs(dx), dy))
                arm_angles.append(arm_angle_deg)

                cv2.putText(frame, f"REL: {int(release_height_score)} pts", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 2)

            # Stride Counter
            if l_ankle.visibility > 0.5 and r_ankle.visibility > 0.5:
                lowest_ankle_y = max(l_ankle.y, r_ankle.y)
                if lowest_ankle_y > 0.82:
                    if not foot_was_down:
                        stride_count += 1
                        foot_was_down = True
                else:
                    foot_was_down = False

            # Velocity
            if l_hip.visibility > 0.5:
                current_hip_x = l_hip.x * process_w
                if prev_hip_x is not None:
                    pixel_dist = abs(current_hip_x - prev_hip_x)
                    vel_px_sec = pixel_dist / time_per_frame
                    velocities.append(vel_px_sec)
                prev_hip_x = current_hip_x

        out.write(frame)
        
        # Clear frame references from memory actively
        del rgb_frame
        del results

    cap.release()
    out.release()
    pose.close()
    
    # Active memory flushing
    gc.collect()

    summary = {
        "strides": stride_count,
        "avg_l_knee": int(np.mean(l_knee_angles)) if l_knee_angles else 0,
        "avg_r_knee": int(np.mean(r_knee_angles)) if r_knee_angles else 0,
        "avg_arm_angle": int(np.mean(arm_angles)) if arm_angles else 0,
        "avg_release_score": round(float(np.mean(release_scores)), 1) if release_scores else 0,
        "avg_velocity": int(np.mean(velocities)) if velocities else 0,
    }

    return True, summary


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_raw.mp4')
    if os.path.exists(input_path):
        os.remove(input_path)
    file.save(input_path)

    epoch_time = int(time.time())
    output_filename = f'analyzed_{epoch_time}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    try:
        success, summary = process_bowling_video(input_path, output_path)
    except Exception as e:
        print(f"Error during runtime processing: {str(e)}")
        success = False

    # Clean up raw upload immediately to free disk/memory space
    if os.path.exists(input_path):
        os.remove(input_path)

    if success:
        return jsonify({
            'video_url': f'/static/{output_filename}',
            'summary': summary
        })
    else:
        return jsonify({'error': 'Server out of memory or processing failed'}), 500


@app.route('/static/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404

    response = make_response(send_file(video_path, mimetype='video/mp4', conditional=True))
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Accept-Ranges'] = 'bytes'
    return response


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
