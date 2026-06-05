import cv2
import mediapipe as mp
import os
import numpy as np
import time
from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)

# Enable cross-origin resource sharing globally for phone testing
CORS(app, resources={r"/*": {"origins": "*"}})

# Define explicit paths for uploads and static analytics assets
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB structural ceiling


def calculate_joint_angle(p1, p2, p3):
    """Calculates the internal vector angle at a specific joint vertex (p2)."""
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def process_bowling_video(video_path, output_path):
    """Core computer vision biomechanics engine with cross-platform H.264 container fallbacks."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, {}

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    time_per_frame = 1.0 / fps

    # WEB-COMPATIBLE FIX: Attempt native H.264 encoding first for absolute web compliance
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    except Exception:
        out = None

    # HARDWARE FALLBACK: Drop down safely to alternative browser-friendly mp4v container layer
    if out is None or not out.isOpened():
        print("⚠️ H.264 avc1 compilation failed/unsupported. Dropping to default mp4v container layer...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    if not out.isOpened():
        print("ERROR: Could not create output video file writer system.")
        return False, {}

    prev_hip_x = None
    stride_count = 0
    foot_was_down = False

    l_knee_angles = []
    r_knee_angles = []
    arm_angles = []
    release_scores = []
    velocities = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
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

            # Left Knee Angle tracking
            if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                l_angle = calculate_joint_angle(l_hip, l_knee, l_ankle)
                l_knee_angles.append(l_angle)
                l_color = (0, 255, 0) if l_angle > 165 else (200, 200, 200)
                cv2.putText(frame, f"L Knee: {int(l_angle)} deg",
                            (int(l_knee.x * w) + 20, int(l_knee.y * h)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, l_color, 3, cv2.LINE_AA)

            # Right Knee Angle tracking
            if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                r_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
                r_knee_angles.append(r_angle)
                r_color = (0, 255, 0) if r_angle > 165 else (200, 200, 200)
                cv2.putText(frame, f"R Knee: {int(r_angle)} deg",
                            (int(r_knee.x * w) + 20, int(r_knee.y * h) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, r_color, 3, cv2.LINE_AA)

            # Arm Release Slant Angle & Elevation
            if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5 and l_shoulder.visibility > 0.5:
                highest_wrist = l_wrist if l_wrist.y < r_wrist.y else r_wrist
                corresponding_shoulder = l_shoulder if highest_wrist == l_wrist else r_shoulder

                wrist_pixel_x = int(highest_wrist.x * w)
                wrist_pixel_y = int(highest_wrist.y * h)

                ground_reference = max(l_ankle.y, r_ankle.y)
                release_height_score = (ground_reference - highest_wrist.y) * 100
                release_scores.append(release_height_score)

                dx = highest_wrist.x - corresponding_shoulder.x
                dy = corresponding_shoulder.y - highest_wrist.y
                arm_angle_deg = np.degrees(np.arctan2(abs(dx), dy))
                arm_angles.append(arm_angle_deg)

                cv2.putText(frame, f"HAND REL: {int(release_height_score)} pts",
                            (wrist_pixel_x + 25, wrist_pixel_y - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 242, 254), 3, cv2.LINE_AA)
                cv2.putText(frame, f"ARM ANGLE: {int(arm_angle_deg)} deg",
                            (wrist_pixel_x + 25, wrist_pixel_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 242, 255), 3, cv2.LINE_AA)

            # Stride Counter Logic
            if l_ankle.visibility > 0.5 and r_ankle.visibility > 0.5:
                lowest_ankle_y = max(l_ankle.y, r_ankle.y)
                if lowest_ankle_y > 0.82:
                    if not foot_was_down:
                        stride_count += 1
                        foot_was_down = True
                else:
                    foot_was_down = False
                cv2.putText(frame, f"Strides Counted: {stride_count}", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

            # Run-up/Delivery Frame Velocity
            if l_hip.visibility > 0.5:
                current_hip_x = l_hip.x * w
                if prev_hip_x is not None:
                    pixel_dist = abs(current_hip_x - prev_hip_x)
                    vel_px_sec = pixel_dist / time_per_frame
                    velocities.append(vel_px_sec)
                    cv2.putText(frame, f"Velocity: {int(vel_px_sec)} px/sec", (40, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 242, 254), 3, cv2.LINE_AA)
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

    # Generate a unique timestamp to break frontend CDN or browser asset caching
    epoch_time = int(time.time())
    output_filename = f'analyzed_{epoch_time}.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    print(f"⏳ Processing dynamic tracking for {output_filename}...")
    success, summary = process_bowling_video(input_path, output_path)
    print("✅ Processing complete!")

    if success:
        return jsonify({
            'video_url': f'/static/{output_filename}',
            'summary': summary
        })
    else:
        return jsonify({'error': 'Video processing failed'}), 500


@app.route('/static/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video asset not found'}), 404

    # Using conditional loop ranges allows direct scrubbing and partial downloads on mobile devices
    return send_file(
        video_path,
        mimetype='video/mp4',
        conditional=True
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
