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

# Job store — tracks async processing state
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


def calculate_elbow_angle(shoulder, elbow, wrist):
    """
    Elbow angle = angle at elbow joint between shoulder-elbow-wrist.
    180 degrees = fully straight arm. Lower = more bent.
    """
    s = np.array([shoulder.x, shoulder.y])
    e = np.array([elbow.x,    elbow.y])
    w = np.array([wrist.x,    wrist.y])
    es = s - e
    ew = w - e
    cos_a = np.dot(es, ew) / (np.linalg.norm(es) * np.linalg.norm(ew) + 1e-6)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def reencode_for_web(raw_path, final_path):
    """ffmpeg re-encode with faststart for browser playback."""
    try:
        result = subprocess.run([
            'ffmpeg', '-y', '-i', raw_path,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
            '-movflags', '+faststart',
            '-an',  # no audio — saves RAM
            final_path
        ], capture_output=True, timeout=180)
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return result.returncode == 0
    except Exception as e:
        print(f"ffmpeg error: {e}")
        if os.path.exists(raw_path):
            try:
                os.rename(raw_path, final_path)
            except Exception:
                pass
        return False


def process_bowling_video(video_path, output_path, job_id):
    """Runs in background thread. Updates jobs dict when done."""
    pose = None
    cap = None
    out = None
    try:
        # Create MediaPipe pose fresh per job — release immediately after
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # LITE model — uses ~3x less RAM than default
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        time_per_frame = 1.0 / fps

        # Downscale if video is very large — saves RAM during processing
        max_dim = 720
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            proc_w = int(orig_w * scale)
            proc_h = int(orig_h * scale)
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
        arm_angles, release_scores, velocities = [], [], []

        # ── ICC Elbow Extension tracking ────────────────────────────
        # We track the elbow angle at two specific moments:
        # 1. When bowling arm reaches horizontal (shoulder level)
        # 2. At the moment of ball release (highest wrist point)
        # Extension = release_angle - horizontal_angle
        # ICC legal limit = 15 degrees
        elbow_at_horizontal   = None   # angle recorded when arm is horizontal
        arm_reached_horizontal = False
        icc_extensions        = []     # list of measured extensions per delivery
        # Phase detection: track wrist Y to find release (peak height = lowest Y)
        prev_wrist_y          = None
        wrist_peak_y          = None
        elbow_at_peak         = None
        bowling_side          = None   # 'left' or 'right' — detected from dominant wrist

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame if needed
            if (proc_w, proc_h) != (orig_w, orig_h):
                frame = cv2.resize(frame, (proc_w, proc_h))

            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            scale     = max(min(w, h) / 720 * 0.5, 0.28)
            thickness = max(1, int(scale * 2))
            pad       = int(scale * 12)

            def put_label(img, text, x, y, color):
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
                x = max(pad, min(x, w - tw - pad * 2))
                y = max(th + pad, min(y, h - pad))
                overlay = img.copy()
                cv2.rectangle(overlay, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad//2), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
                cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

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
                l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                r_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                l_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
                r_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]

                if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                    l_angle = calculate_joint_angle(l_hip, l_knee, l_ankle)
                    l_knee_angles.append(l_angle)
                    col = (0,255,0) if l_angle > 165 else (200,200,200)
                    put_label(frame, f"L {int(l_angle)}\u00b0",
                              int(l_knee.x*w)+12, int(l_knee.y*h), col)

                if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                    r_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
                    r_knee_angles.append(r_angle)
                    col = (0,255,0) if r_angle > 165 else (200,200,200)
                    put_label(frame, f"R {int(r_angle)}\u00b0",
                              int(r_knee.x*w)+12, int(r_knee.y*h)-20, col)

                # ── ICC Elbow Extension Measurement ─────────────────────────
                # Determine bowling arm (side with higher wrist = lower Y value)
                both_wrists_visible = (l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5
                                       and l_shoulder.visibility > 0.5 and r_shoulder.visibility > 0.5
                                       and l_elbow.visibility > 0.5 and r_elbow.visibility > 0.5)

                if both_wrists_visible:
                    # Bowling arm = whichever wrist is higher in the frame (lower Y)
                    if bowling_side is None:
                        bowling_side = 'left' if l_wrist.y < r_wrist.y else 'right'

                    if bowling_side == 'left':
                        b_shoulder, b_elbow, b_wrist = l_shoulder, l_elbow, l_wrist
                    else:
                        b_shoulder, b_elbow, b_wrist = r_shoulder, r_elbow, r_wrist

                    current_elbow_angle = calculate_elbow_angle(b_shoulder, b_elbow, b_wrist)
                    current_wrist_y     = b_wrist.y  # lower Y = higher in frame

                    # PHASE 1: Detect when arm reaches horizontal
                    # Horizontal = elbow and shoulder at same height (Y within tolerance)
                    arm_is_horizontal = abs(b_shoulder.y - b_elbow.y) < 0.12

                    if arm_is_horizontal and not arm_reached_horizontal:
                        # Capture elbow angle at horizontal position
                        elbow_at_horizontal    = current_elbow_angle
                        wrist_peak_y           = current_wrist_y
                        elbow_at_peak          = current_elbow_angle
                        arm_reached_horizontal = True

                    # PHASE 2: Track wrist rising (Y decreasing) to find release peak
                    if arm_reached_horizontal:
                        if current_wrist_y < (wrist_peak_y or 1.0):
                            wrist_peak_y  = current_wrist_y
                            elbow_at_peak = current_elbow_angle

                        # PHASE 3: Wrist starts descending = release moment just passed
                        if (prev_wrist_y is not None
                                and current_wrist_y > (prev_wrist_y + 0.02)
                                and elbow_at_horizontal is not None
                                and elbow_at_peak is not None):

                            # ICC extension = how much elbow STRAIGHTENED from horizontal to peak
                            extension = max(0.0, elbow_at_peak - elbow_at_horizontal)
                            icc_extensions.append(extension)
                            arm_angles.append(elbow_at_peak)

                            # Overlay at wrist position
                            wx = int(b_wrist.x * w)
                            wy = int(b_wrist.y * h)
                            ext_color = (0,255,0) if extension <= 15 else (0,0,255)
                            put_label(frame, f"ELBOW EXT: {int(extension)}\u00b0",
                                      wx + 14, wy - 30, ext_color)
                            legal_txt = "LEGAL" if extension <= 15 else "ILLEGAL ACTION!"
                            put_label(frame, legal_txt, wx + 14, wy - 6, ext_color)

                            # Reset for next delivery
                            arm_reached_horizontal = False
                            elbow_at_horizontal    = None
                            wrist_peak_y           = None
                            elbow_at_peak          = None

                    # Always show current live elbow angle
                    ex = int(b_elbow.x * w)
                    ey = int(b_elbow.y * h)
                    put_label(frame, f"Elbow: {int(current_elbow_angle)}\u00b0",
                              ex + 10, ey, (255, 220, 0))

                    prev_wrist_y = current_wrist_y

                    # Release height score (kept for dashboard)
                    ground_ref = max(l_ankle.y, r_ankle.y)
                    rel_score  = (ground_ref - b_wrist.y) * 100
                    release_scores.append(rel_score)

                line_h = int(scale * 36)
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

        # Close MediaPipe immediately to free RAM before ffmpeg
        pose.close()
        pose = None

        reencode_for_web(raw_output, output_path)

        # Clean up input file to save disk
        try:
            os.remove(video_path)
        except Exception:
            pass

        summary = {
            "strides":           stride_count,
            "avg_l_knee":        int(np.mean(l_knee_angles))    if l_knee_angles   else 0,
            "avg_r_knee":        int(np.mean(r_knee_angles))    if r_knee_angles   else 0,
            "avg_elbow_ext":     round(float(np.mean(icc_extensions)), 1) if icc_extensions else None,
            "max_elbow_ext":     round(float(np.max(icc_extensions)),  1) if icc_extensions else None,
            "icc_legal":         bool(np.max(icc_extensions) <= 15)       if icc_extensions else None,
            "avg_release_score": round(float(np.mean(release_scores)), 1) if release_scores else 0,
            "avg_velocity":      int(np.mean(velocities))        if velocities      else 0,
        }

        with jobs_lock:
            jobs[job_id] = {
                'status': 'done',
                'video_url': f'/static/{os.path.basename(output_path)}',
                'summary': summary
            }
        print(f"Job {job_id} done.")

    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        # Clean up on failure
        for path in [video_path]:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        with jobs_lock:
            jobs[job_id] = {'status': 'error', 'error': str(e)}
    finally:
        # Always release resources
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
    """Accept upload, start background thread, return job_id immediately."""
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

    # Clean up old analyzed files to save disk on free tier
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

    print(f"Job {job_id} started.")
    return jsonify({'job_id': job_id}), 202


@app.route('/status/<job_id>')
def job_status(job_id):
    """Frontend polls this every 3 seconds."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(job)


@app.route('/static/<filename>')
def serve_video(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, mimetype='video/mp4', conditional=True)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
