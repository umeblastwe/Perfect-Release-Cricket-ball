import cv2
import mediapipe as mp
import os
import subprocess
import numpy as np
import time
import threading
import uuid
from flask import Flask, request, render_template, jsonify, send_file

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'static')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# Global dictionary to track async job statuses and summaries
JOBS = {}

def calculate_joint_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def reencode_for_web(input_path, output_path):
    try:
        tmp_path = input_path.replace('.mp4', '_tmp.mp4')
        os.rename(input_path, tmp_path)
        result = subprocess.run([
            'ffmpeg', '-y',
            '-i', tmp_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',  # 'ultrafast' saves CPU & prevents timeouts on Render
            '-crf', '28',            # Slightly lower quality saves huge RAM/Disk space
            '-c:a', 'aac',
            '-movflags', '+faststart',
            output_path
        ], capture_output=True, timeout=180)
        
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        if result.returncode == 0:
            print("✅ ffmpeg re-encode successful")
            return True
        else:
            print(f"⚠️ ffmpeg failed: {result.stderr.decode()}")
            return False
    except Exception as e:
        print(f"⚠️ ffmpeg failed: {e}")
        return False


def process_bowling_video_async(job_id, video_path, output_filename):
    """
    Runs in a background thread. Updates the global JOBS dictionary.
    """
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    raw_output = output_path.replace('.mp4', '_raw.mp4')
    
    mp_pose = mp.solutions.pose
    
    # CRITICAL RAM OPTIMIZATION: 
    # Use context manager 'with' to ensure MediaPipe is completely wiped from RAM after use.
    # model_complexity=0 drastically reduces memory usage on Render 512MB tier.
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # 0 = Light, 1 = Full, 2 = Heavy. 0 is mandatory for low RAM!
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        mp_drawing = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            JOBS[job_id] = {"status": "failed", "error": "Could not open video file"}
            return

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 25.0
        time_per_frame = 1.0 / fps

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Safer fallback default for headless servers
        out = cv2.VideoWriter(raw_output, fourcc, fps, (orig_w, orig_h))

        if not out.isOpened():
            cap.release()
            JOBS[job_id] = {"status": "failed", "error": "Could not initialize video writer"}
            return

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
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                # Left Knee
                if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                    l_angle = calculate_joint_angle(l_hip, l_knee, l_ankle)
                    l_knee_angles.append(l_angle)
                    l_color = (0, 255, 0) if l_angle > 165 else (200, 200, 200)
                    put_label(frame, f"L {int(l_angle)}\u00b0", int(l_knee.x * w) + 12, int(l_knee.y * h), l_color)

                # Right Knee
                if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                    r_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
                    r_knee_angles.append(r_angle)
                    r_color = (0, 255, 0) if r_angle > 165 else (200, 200, 200)
                    put_label(frame, f"R {int(r_angle)}\u00b0", int(r_knee.x * w) + 12, int(r_knee.y * h) - 20, r_color)

                # Arm & Release
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

                    put_label(frame, f"REL {int(release_height_score)}", wrist_pixel_x + 14, wrist_pixel_y - 30, (0, 242, 254))
                    put_label(frame, f"ARM {int(arm_angle_deg)}\u00b0", wrist_pixel_x + 14, wrist_pixel_y - 6, (255, 220, 0))

                # Strides & Velocity
                line_h = int(scale * 38)
                if l_ankle.visibility > 0.5 and r_ankle.visibility > 0.5:
                    lowest_ankle_y = max(l_ankle.y, r_ankle.y)
                    if lowest_ankle_y > 0.82:
                        if not foot_was_down:
                            stride_count += 1
                            foot_was_down = True
                    else:
                        foot_was_down = False
                    put_label(frame, f"Strides: {stride_count}", 16, line_h, (255, 255, 255))

                if l_hip.visibility > 0.5:
                    current_hip_x = l_hip.x * w
                    if prev_hip_x is not None:
                        pixel_dist = abs(current_hip_x - prev_hip_x)
                        vel_px_sec = pixel_dist / time_per_frame
                        velocities.append(vel_px_sec)
                        put_label(frame, f"Vel: {int(vel_px_sec)} px/s", 16, line_h * 2 + 4, (0, 242, 254))
                    prev_hip_x = current_hip_x

            out.write(frame)

        cap.release()
        out.release()

    # Context closed. MediaPipe is deleted from RAM here. 
    # Now trigger ffmpeg web compression
    reencoded = reencode_for_web(raw_output, output_path)
    if not reencoded and os.path.exists(raw_output):
        os.rename(raw_output, output_path)

    # Clean up raw source video to save disk space
    if os.path.exists(video_path):
        os.remove(video_path)

    summary = {
        "strides": stride_count,
        "avg_l_knee": int(np.mean(l_knee_angles)) if l_knee_angles else 0,
        "avg_r_knee": int(np.mean(r_knee_angles)) if r_knee_angles else 0,
        "avg_arm_angle": int(np.mean(arm_angles)) if arm_angles else 0,
        "avg_release_score": round(float(np.mean(release_scores)), 1) if release_scores else 0,
        "avg_velocity": int(np.mean(velocities)) if velocities else 0,
    }

    # Mark the job as complete
    JOBS[job_id] = {
        "status": "completed",
        "video_url": f"/static/{output_filename}",
        "summary": summary
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    job_id = str(uuid.uuid4())
    input_filename = f'input_{job_id}.mp4'
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    file.save(input_path)

    output_filename = f'analyzed_{job_id}.mp4'

    # Auto clean older processed files before starting a new run
    for f in os.listdir(app.config['OUTPUT_FOLDER']):
        if f.startswith('analyzed_'):
            try:
                os.remove(os.path.join(app.config['OUTPUT_FOLDER'], f))
            except Exception:
                pass

    # Initialize Job State
    JOBS[job_id] = {"status": "processing"}

    # Fire and forget: process in background thread
    thread = threading.Thread(target=process_bowling_video_async, args=(job_id, input_path, output_filename))
    thread.daemon = True
    thread.start()

    # Instantly reply to frontend with the job_id so gunicorn worker doesn't time out
    return jsonify({'job_id': job_id, 'status': 'processing'}), 202


@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """
    Frontend should poll this route every 2-3 seconds using JavaScript setInterval.
    """
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


@app.route('/static/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        video_path,
        mimetype='video/mp4',
        conditional=True
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
