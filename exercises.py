# exercises.py
# Each exercise now includes form_checks for real-time feedback

EXERCISES = {
    "1": {
        "name":           "Bicep Curl",
        "joint_a":        "left_shoulder",
        "joint_b":        "left_elbow",
        "joint_c":        "left_wrist",
        "up_threshold":   50,
        "down_threshold": 160,
        "direction":      "up_first",
        "tip":            "Keep your elbow locked at your side",
        "form_checks": [
            {
                "type":    "angle_range",
                "joint_a": "left_shoulder",
                "joint_b": "left_elbow",
                "joint_c": "left_wrist",
                "min":     10,
                "max":     170,
                "message": "⚠️ Full range of motion — curl all the way up!"
            },
            {
                "type":    "symmetry",
                "point_a": "left_shoulder",
                "point_b": "right_shoulder",
                "axis":    "y",
                "tolerance": 30,
                "message": "⚠️ Keep your shoulders level!"
            }
        ]
    },
    "2": {
        "name":           "Squat",
        "joint_a":        "left_hip",
        "joint_b":        "left_knee",
        "joint_c":        "left_ankle",
        "up_threshold":   90,
        "down_threshold": 160,
        "direction":      "down_first",
        "tip":            "Keep your knees behind your toes",
        "form_checks": [
            {
                "type":    "angle_range",
                "joint_a": "left_hip",
                "joint_b": "left_knee",
                "joint_c": "left_ankle",
                "min":     60,
                "max":     175,
                "message": "⚠️ Don't cave your knees inward!"
            },
            {
                "type":    "symmetry",
                "point_a": "left_hip",
                "point_b": "right_hip",
                "axis":    "y",
                "tolerance": 25,
                "message": "⚠️ Keep your hips level — don't lean!"
            }
        ]
    },
    "3": {
        "name":           "Push-up",
        "joint_a":        "left_shoulder",
        "joint_b":        "left_elbow",
        "joint_c":        "left_wrist",
        "up_threshold":   90,
        "down_threshold": 155,
        "direction":      "down_first",
        "tip":            "Keep your body in a straight line",
        "form_checks": [
            {
                "type":    "angle_range",
                "joint_a": "left_shoulder",
                "joint_b": "left_hip",
                "joint_c": "left_knee",
                "min":     160,
                "max":     195,
                "message": "⚠️ Keep your body straight — don't sag!"
            },
            {
                "type":    "symmetry",
                "point_a": "left_shoulder",
                "point_b": "right_shoulder",
                "axis":    "y",
                "tolerance": 20,
                "message": "⚠️ Keep your shoulders even!"
            }
        ]
    },
    "4": {
        "name":           "Shoulder Press",
        "joint_a":        "left_elbow",
        "joint_b":        "left_shoulder",
        "joint_c":        "left_hip",
        "up_threshold":   160,
        "down_threshold": 50,
        "direction":      "up_first",
        "tip":            "Don't arch your lower back",
        "form_checks": [
            {
                "type":    "angle_range",
                "joint_a": "left_shoulder",
                "joint_b": "left_hip",
                "joint_c": "left_knee",
                "min":     150,
                "max":     195,
                "message": "⚠️ Don't arch your back!"
            },
            {
                "type":    "symmetry",
                "point_a": "left_elbow",
                "point_b": "right_elbow",
                "axis":    "y",
                "tolerance": 30,
                "message": "⚠️ Press both arms evenly!"
            }
        ]
    },
}