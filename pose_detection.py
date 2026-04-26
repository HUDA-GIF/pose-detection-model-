import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from exercises import EXERCISES
from collections import deque
import time

# ── 1. Model path ─────────────────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_full.task"

# ── 2. Optimization settings ──────────────────────────────────────────────
SMOOTHING_WINDOW  = 5     # average over last 5 angle readings
REP_COOLDOWN_SECS = 1.0   # seconds to wait after counting a rep
MIN_VISIBILITY    = 0.6   # stricter visibility threshold (was 0.5)

# ── 3. Landmark indices ───────────────────────────────────────────────────
LANDMARKS = {
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28
}

# ── 4. Angle smoother class ───────────────────────────────────────────────
class AngleSmoother:
    """
    Keeps a rolling window of recent angle values.
    Returns the average — eliminates jitter.
    """
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)

    def update(self, angle):
        self.window.append(angle)
        return round(sum(self.window) / len(self.window), 1)

    def reset(self):
        self.window.clear()

# ── 5. FPS counter class ──────────────────────────────────────────────────
class FPSCounter:
    """Calculates real-time frames per second"""
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0

    def update(self):
        now = time.time()
        self.fps = round(1.0 / (now - self.prev_time + 1e-6), 1)
        self.prev_time = now
        return self.fps

# ── 6. Core functions ─────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return round(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), 1)

def extract_keypoints(detection_result, w, h):
    keypoints = {}
    if not detection_result.pose_landmarks:
        return keypoints
    landmarks = detection_result.pose_landmarks[0]
    for name, idx in LANDMARKS.items():
        lm = landmarks[idx]
        keypoints[name] = (int(lm.x * w), int(lm.y * h), lm.visibility)
    return keypoints

def xy(kp, name):
    x, y, _ = kp[name]
    return (x, y)

def visible(kp, *names):
    # Uses stricter MIN_VISIBILITY threshold
    return all(kp.get(n, (0,0,0))[2] > MIN_VISIBILITY for n in names)

def draw_landmarks_on_image(image, detection_result):
    if not detection_result.pose_landmarks:
        return image
    for pose_landmarks in detection_result.pose_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            image, proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return image

# ── 7. Optimized rep counter ──────────────────────────────────────────────
def update_rep_counter(angle, stage, reps, exercise, last_rep_time):
    """
    Rep counter with cooldown to prevent double counting.
    Won't count a new rep until REP_COOLDOWN_SECS has passed.
    """
    direction   = exercise["direction"]
    now         = time.time()
    cooldown_ok = (now - last_rep_time) > REP_COOLDOWN_SECS

    if direction == "up_first":
        if angle < exercise["up_threshold"]:
            stage = "UP"
        if angle > exercise["down_threshold"] and stage == "UP" and cooldown_ok:
            stage         = "DOWN"
            reps         += 1
            last_rep_time = now
            print(f"\n  ✅ Rep {reps}!")

    elif direction == "down_first":
        if angle < exercise["up_threshold"]:
            stage = "DOWN"
        if angle > exercise["down_threshold"] and stage == "DOWN" and cooldown_ok:
            stage         = "UP"
            reps         += 1
            last_rep_time = now
            print(f"\n  ✅ Rep {reps}!")

    return stage, reps, last_rep_time

# ── 8. Form checker ───────────────────────────────────────────────────────
def check_form(kp, exercise):
    feedback = []
    for check in exercise.get("form_checks", []):
        if check["type"] == "angle_range":
            ja, jb, jc = check["joint_a"], check["joint_b"], check["joint_c"]
            if visible(kp, ja, jb, jc):
                angle = calculate_angle(xy(kp, ja), xy(kp, jb), xy(kp, jc))
                if not (check["min"] <= angle <= check["max"]):
                    feedback.append(check["message"])
        elif check["type"] == "symmetry":
            pa, pb = check["point_a"], check["point_b"]
            if visible(kp, pa, pb):
                idx   = 0 if check["axis"] == "x" else 1
                if abs(kp[pa][idx] - kp[pb][idx]) > check["tolerance"]:
                    feedback.append(check["message"])
    return feedback

# ── 9. Draw UI ────────────────────────────────────────────────────────────
def draw_ui(frame, reps, stage, angle, exercise, feedback, fps):
    h, w = frame.shape[:2]

    # Left panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (230, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, exercise["name"], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "REPS", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, str(reps), (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 255, 0), 3)
    cv2.putText(frame, "STAGE", (10, 148),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    color = (0, 255, 0) if stage in ["DOWN", "UP"] else (100, 100, 100)
    cv2.putText(frame, stage if stage else "---", (10, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Angle + FPS top right
    angle_text = f"Angle: {angle}°" if angle else "Angle: ---"
    cv2.putText(frame, angle_text, (w - 200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # FPS indicator — color coded
    fps_color = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps}", (w - 120, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)

    # Form feedback
    if feedback:
        for i, msg in enumerate(feedback[:3]):
            y_pos     = 70 + (i * 35)
            text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            cv2.rectangle(frame,
                         (w//2 - 10, y_pos - 22),
                         (w//2 + text_size[0] + 10, y_pos + 8),
                         (0, 0, 180), -1)
            cv2.putText(frame, msg, (w//2, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    else:
        if stage in ["UP", "DOWN"]:
            cv2.putText(frame, "✓ Perfect Form!", (w//2 - 80, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Tip
    cv2.putText(frame, f"Tip: {exercise['tip']}", (10, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return frame

# ── 10. Menu ──────────────────────────────────────────────────────────────
def show_menu():
    print("\n" + "="*40)
    print("   🏋️  AI FITNESS COACH  v2.0")
    print("="*40)
    for key, ex in EXERCISES.items():
        print(f"   {key}. {ex['name']}")
    print("="*40)
    choice = input("Select exercise (1-4): ").strip()
    return EXERCISES.get(choice, EXERCISES["1"])

# ── 11. Configure model ───────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

# ── 12. Setup ─────────────────────────────────────────────────────────────
exercise      = show_menu()
reps          = 0
stage         = None
last_rep_time = 0
smoother      = AngleSmoother(window_size=SMOOTHING_WINDOW)
fps_counter   = FPSCounter()

print(f"\n▶ Starting: {exercise['name']}  (optimized v2.0)")
print("  Press R to reset | Press Q to quit\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

# ── 13. Main loop ─────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w  = frame.shape[:2]
    fps   = fps_counter.update()
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    annotated = draw_landmarks_on_image(rgb.copy(), result)
    display   = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    kp       = extract_keypoints(result, w, h)
    angle    = None
    feedback = []

    if kp and visible(kp, exercise["joint_a"],
                          exercise["joint_b"],
                          exercise["joint_c"]):

        # Raw angle
        raw_angle = calculate_angle(
            xy(kp, exercise["joint_a"]),
            xy(kp, exercise["joint_b"]),
            xy(kp, exercise["joint_c"])
        )

        # ── Smoothed angle ─────────────────────────────────────────────
        angle = smoother.update(raw_angle)

        # ── Rep counting with cooldown ─────────────────────────────────
        stage, reps, last_rep_time = update_rep_counter(
            angle, stage, reps, exercise, last_rep_time
        )

        # ── Form check ─────────────────────────────────────────────────
        feedback = check_form(kp, exercise)

        # Draw angle at joint
        jx, jy, _ = kp[exercise["joint_b"]]
        cv2.putText(display, f"{angle}°", (jx - 40, jy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        print(f"  Angle: {angle}° | Reps: {reps} | FPS: {fps}   ", end="\r")

    display = draw_ui(display, reps, stage, angle, exercise, feedback, fps)
    cv2.imshow(f"AI Fitness Coach v2.0 — {exercise['name']}", display)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reps, stage, last_rep_time = 0, None, 0
        smoother.reset()
        print("\n🔄 Counter reset!")

# ── 14. Clean up ──────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
detector.close()
print(f"\n✅ Session done! {exercise['name']} — Total reps: {reps}")