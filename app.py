# app.py — Flask web server for AI Fitness Coach
from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from collections import deque
import time
import threading
from exercises import EXERCISES

app = Flask(__name__)

# ── 1. Optimization settings ──────────────────────────────────────────────
MODEL_PATH        = "pose_landmarker_full.task"
SMOOTHING_WINDOW  = 5
REP_COOLDOWN_SECS = 1.0
MIN_VISIBILITY    = 0.6

LANDMARKS = {
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28
}

# ── 2. Shared app state (thread-safe) ─────────────────────────────────────
state = {
    "reps":          0,
    "stage":         None,
    "angle":         None,
    "feedback":      [],
    "fps":           0,
    "exercise_key":  "1",
    "last_rep_time": 0,
    "smoother":      deque(maxlen=SMOOTHING_WINDOW)
}
state_lock = threading.Lock()

# ── 3. Core functions ─────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cosine  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
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
    return kp[name][:2]

def visible(kp, *names):
    return all(kp.get(n, (0,0,0))[2] > MIN_VISIBILITY for n in names)

def smooth_angle(window, angle):
    window.append(angle)
    return round(sum(window) / len(window), 1)

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

def update_rep_counter(angle, stage, reps, exercise, last_rep_time):
    direction   = exercise["direction"]
    now         = time.time()
    cooldown_ok = (now - last_rep_time) > REP_COOLDOWN_SECS
    if direction == "up_first":
        if angle < exercise["up_threshold"]:
            stage = "UP"
        if angle > exercise["down_threshold"] and stage == "UP" and cooldown_ok:
            stage = "DOWN"; reps += 1; last_rep_time = now
    elif direction == "down_first":
        if angle < exercise["up_threshold"]:
            stage = "DOWN"
        if angle > exercise["down_threshold"] and stage == "DOWN" and cooldown_ok:
            stage = "UP"; reps += 1; last_rep_time = now
    return stage, reps, last_rep_time

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
                idx = 0 if check["axis"] == "x" else 1
                if abs(kp[pa][idx] - kp[pb][idx]) > check["tolerance"]:
                    feedback.append(check["message"])
    return feedback

# ── 4. Camera thread ──────────────────────────────────────────────────────
def camera_thread():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    cap      = cv2.VideoCapture(0)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        now  = time.time()
        fps  = round(1.0 / (now - prev_time + 1e-6), 1)
        prev_time = now

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        annotated = draw_landmarks_on_image(rgb.copy(), result)
        display   = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        with state_lock:
            exercise = EXERCISES[state["exercise_key"]]
            kp       = extract_keypoints(result, w, h)

            if kp and visible(kp, exercise["joint_a"],
                                  exercise["joint_b"],
                                  exercise["joint_c"]):

                raw   = calculate_angle(
                    xy(kp, exercise["joint_a"]),
                    xy(kp, exercise["joint_b"]),
                    xy(kp, exercise["joint_c"])
                )
                angle = smooth_angle(state["smoother"], raw)

                state["stage"], state["reps"], state["last_rep_time"] = \
                    update_rep_counter(
                        angle, state["stage"], state["reps"],
                        exercise, state["last_rep_time"]
                    )

                state["feedback"] = check_form(kp, exercise)
                state["angle"]    = angle

                # Draw angle
                jx, jy, _ = kp[exercise["joint_b"]]
                cv2.putText(display, f"{angle}°", (jx-40, jy-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            state["fps"] = fps

        # Encode frame as JPEG for streaming
        _, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ── 5. HTML page ──────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Fitness Coach</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { background:#111; color:#fff; font-family:Arial,sans-serif; }
        h1   { text-align:center; padding:15px; color:#0f0; font-size:1.4rem; }
        .container { display:flex; justify-content:center; gap:20px; padding:10px; flex-wrap:wrap; }
        .feed  { position:relative; }
        img    { width:640px; max-width:100%; border:2px solid #0f0; border-radius:8px; }
        .panel { background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:20px; width:220px; }
        .stat  { margin-bottom:18px; }
        .label { font-size:0.75rem; color:#aaa; text-transform:uppercase; }
        .value { font-size:2.5rem; color:#0f0; font-weight:bold; }
        .stage { font-size:1.2rem; color:#ff0; }
        .angle { font-size:1rem; color:#0ff; }
        .fps   { font-size:0.85rem; color:#0f0; }
        .feedback { background:#300; border:1px solid #f00; border-radius:6px;
                    padding:10px; margin-top:10px; font-size:0.8rem; color:#f88; }
        .good { background:#030; border:1px solid #0f0; color:#0f0; }
        select, button { width:100%; padding:8px; margin-top:8px; border-radius:6px;
                         border:1px solid #555; background:#222; color:#fff; cursor:pointer; }
        button { background:#1a4a1a; color:#0f0; border-color:#0f0; }
        button:hover { background:#0f0; color:#000; }
    </style>
</head>
<body>
    <h1>🏋️ AI Fitness Coach</h1>
    <div class="container">
        <div class="feed">
            <img src="/video_feed" />
        </div>
        <div class="panel">
            <div class="stat">
                <div class="label">Exercise</div>
                <select id="exercise" onchange="switchExercise(this.value)">
                    {% for key, ex in exercises.items() %}
                    <option value="{{ key }}">{{ ex.name }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="stat">
                <div class="label">Reps</div>
                <div class="value" id="reps">0</div>
            </div>
            <div class="stat">
                <div class="label">Stage</div>
                <div class="stage" id="stage">---</div>
            </div>
            <div class="stat">
                <div class="label">Angle</div>
                <div class="angle" id="angle">---</div>
            </div>
            <div class="stat">
                <div class="fps" id="fps">FPS: ---</div>
            </div>
            <div id="feedback" class="feedback good">Waiting for pose...</div>
            <button onclick="resetCounter()">🔄 Reset Counter</button>
        </div>
    </div>
    <script>
        // Poll stats every 300ms
        setInterval(() => {
            fetch('/stats').then(r => r.json()).then(data => {
                document.getElementById('reps').textContent  = data.reps;
                document.getElementById('stage').textContent = data.stage || '---';
                document.getElementById('angle').textContent = data.angle ? data.angle + '°' : '---';
                document.getElementById('fps').textContent   = 'FPS: ' + data.fps;
                const fb = document.getElementById('feedback');
                if (data.feedback.length > 0) {
                    fb.className   = 'feedback';
                    fb.textContent = data.feedback.join(' | ');
                } else {
                    fb.className   = 'feedback good';
                    fb.textContent = data.stage ? '✓ Perfect Form!' : 'Waiting for pose...';
                }
            });
        }, 300);

        function switchExercise(key) {
            fetch('/switch/' + key, {method:'POST'});
        }

        function resetCounter() {
            fetch('/reset', {method:'POST'});
        }
    </script>
</body>
</html>
"""

# ── 6. Routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML, exercises=EXERCISES)

@app.route('/video_feed')
def video_feed():
    return Response(camera_thread(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with state_lock:
        return jsonify({
            "reps":     state["reps"],
            "stage":    state["stage"],
            "angle":    state["angle"],
            "feedback": state["feedback"],
            "fps":      state["fps"]
        })

@app.route('/switch/<key>', methods=['POST'])
def switch_exercise(key):
    with state_lock:
        if key in EXERCISES:
            state["exercise_key"]  = key
            state["reps"]          = 0
            state["stage"]         = None
            state["angle"]         = None
            state["last_rep_time"] = 0
            state["smoother"].clear()
    return jsonify({"status": "ok"})

@app.route('/reset', methods=['POST'])
def reset():
    with state_lock:
        state["reps"]          = 0
        state["stage"]         = None
        state["last_rep_time"] = 0
        state["smoother"].clear()
    return jsonify({"status": "ok"})

# ── 7. Run server ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀 AI Fitness Coach Web App")
    print("   Open your browser at: http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=False, threaded=True)