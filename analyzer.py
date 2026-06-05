import cv2
import mediapipe as mp
import os
import sys
import numpy as np

# =====================================================================
# CONFIGURATION PANEL - CHANGE YOUR SETTINGS HERE
# =====================================================================
CAMERA_VIEW = "side"  # Set to "side" or "front"
# =====================================================================

def find_any_video_file():
    video_extensions = ('.mp4', '.mov', '.avi', '.m4v', '.MOV', '.MP4')
    current_folder = os.getcwd()
    for file in os.listdir(current_folder):
        if file.endswith(video_extensions) and "analyzed_output" not in file:
            return file
    return None

def calculate_joint_angle(p1, p2, p3):
    """Calculates the angle at the vertex (p2) using three points."""
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y]) 
    c = np.array([p3.x, p3.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def run_perfect_release_ai():
    video_file = find_any_video_file()
    if video_file is None:
        print("❌ Error: No video file found in this folder!")
        sys.exit()
        
    print(f"🎯 PROCESSING VIDEO: {video_file} | MODE: {CAMERA_VIEW.upper()} VIEW")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"❌ Error: Can't open video file {video_file}")
        return

    # Grab unscaled video resolution parameters directly from the source clip
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1.0 / fps if fps > 0 else 0.0167
    
    # Configure video file rendering destination matching original resolution dimensions
    output_filename = os.path.join(os.getcwd(), "analyzed_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  
    out = cv2.VideoWriter(output_filename, fourcc, fps if fps > 0 else 25.0, (orig_w, orig_h))

    prev_hip_x = None
    stride_count = 0
    foot_was_down = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw standard skeleton dots and connections
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 242, 254), thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            landmarks = results.pose_landmarks.landmark
            head = landmarks[mp_pose.PoseLandmark.NOSE]
            
            # Extract Lower Body
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Extract Upper Body (Shoulders and Wrists)
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # -----------------------------------------------------------------
            # RUN ANALYSIS BASED ON SELECTED CAMERA VIEW
            # -----------------------------------------------------------------
            if CAMERA_VIEW.lower() == "side":
                # --- METRIC 1: DUAL KNEE BRACE CHECK (LARGE BOLD FONT) ---
                if l_hip.visibility > 0.5 and l_knee.visibility > 0.5 and l_ankle.visibility > 0.5:
                    l_angle = calculate_joint_angle(l_hip, l_knee, l_ankle)
                    l_color = (0, 255, 0) if l_angle > 165 else (200, 200, 200)
                    cv2.putText(frame, f"L Knee: {int(l_angle)} deg", (int(l_knee.x * w) + 20, int(l_knee.y * h)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, l_color, 3, cv2.LINE_AA)

                if r_hip.visibility > 0.5 and r_knee.visibility > 0.5 and r_ankle.visibility > 0.5:
                    r_angle = calculate_joint_angle(r_hip, r_knee, r_ankle)
                    r_color = (0, 255, 0) if r_angle > 165 else (200, 200, 200)
                    cv2.putText(frame, f"R Knee: {int(r_angle)} deg", (int(r_knee.x * w) + 20, int(r_knee.y * h) - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, r_color, 3, cv2.LINE_AA)

                # --- METRIC 2: DUAL HAND TRACKING (HEIGHT + GEOMETRIC ANGLE) ---
                if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5 and l_shoulder.visibility > 0.5:
                    # A) Find the bowling hand (the highest wrist on screen)
                    highest_wrist = l_wrist if l_wrist.y < r_wrist.y else r_wrist
                    corresponding_shoulder = l_shoulder if highest_wrist == l_wrist else r_shoulder
                    
                    wrist_pixel_x = int(highest_wrist.x * w)
                    wrist_pixel_y = int(highest_wrist.y * h)
                    
                    # B) Calculate Vertical Height Percentage Score
                    ground_reference = max(l_ankle.y, r_ankle.y)
                    release_height_score = (ground_reference - highest_wrist.y) * 100
                    
                    # C) Calculate Physical Arm Tilt Angle relative to a straight vertical line
                    dx = highest_wrist.x - corresponding_shoulder.x
                    dy = corresponding_shoulder.y - highest_wrist.y
                    arm_angle_deg = np.degrees(np.arctan2(abs(dx), dy))
                    
                    # D) Overlay BOTH values onto the screen with large text lines
                    cv2.putText(frame, f"HAND REL: {int(release_height_score)} pts", 
                                (wrist_pixel_x + 25, wrist_pixel_y - 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 242, 254), 3, cv2.LINE_AA)
                                
                    cv2.putText(frame, f"ARM ANGLE: {int(arm_angle_deg)} deg", 
                                (wrist_pixel_x + 25, wrist_pixel_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 242, 0), 3, cv2.LINE_AA)

                # --- METRIC 3: STRIDE AND SPEED SCOREBOARD ---
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

                if l_hip.visibility > 0.5:
                    current_hip_x = l_hip.x * w
                    if prev_hip_x is not None:
                        pixel_dist = abs(current_hip_x - prev_hip_x)
                        vel_px_sec = pixel_dist / time_per_frame
                        cv2.putText(frame, f"Velocity: {int(vel_px_sec)} px/sec", (40, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 242, 254), 3, cv2.LINE_AA)
                    prev_hip_x = current_hip_x

            elif CAMERA_VIEW.lower() == "front":
                # --- FRONT-ON METRIC: LATERAL TRUNK TILT ---
                if l_shoulder.visibility > 0.5 and r_shoulder.visibility > 0.5 and l_hip.visibility > 0.5:
                    mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
                    mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
                    mid_hip_x = (l_hip.x + r_hip.x) / 2
                    mid_hip_y = (l_hip.y + r_hip.y) / 2
                    
                    dx = mid_shoulder_x - mid_hip_x
                    dy = mid_hip_y - mid_shoulder_y
                    trunk_tilt = np.degrees(np.arctan2(abs(dx), dy))
                    
                    tilt_color = (0, 255, 0) if trunk_tilt < 30 else (0, 165, 255)
                    cv2.putText(frame, f"Trunk Tilt: {int(trunk_tilt)} deg", (40, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, tilt_color, 3, cv2.LINE_AA)

        # Write out every tracked frame to video file at native resolution
        out.write(frame)

        cv2.imshow("Perfect Release Ultimate AI Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"🏁 Processing complete. File saved to: {output_filename}")

run_perfect_release_ai()
