# 🏋️ AI Fitness Coach — FYP

Real-time AI-powered fitness coaching system built with MediaPipe, OpenCV, and Flask.

## Features
- Real-time pose detection using Google MediaPipe BlazePose
- Rep counting for 4 exercises
- Form correction with real-time feedback
- Web dashboard via Flask

## Exercises Supported
| Exercise | Joint Tracked |
|---|---|
| Bicep Curl | Elbow angle |
| Squat | Knee angle |
| Push-up | Elbow angle |
| Shoulder Press | Shoulder angle |

## API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web dashboard |
| `/video_feed` | GET | Live camera stream |
| `/stats` | GET | Current reps, stage, angle, feedback |
| `/switch/<key>` | POST | Switch exercise (1-4) |
| `/reset` | POST | Reset rep counter |

## Setup (Python 3.11 required)
```bash
git clone https://github.com/HUDA-GIF/ai-fitness-coach.git
cd ai-fitness-coach
py -3.11 -m venv venv
venv\Scripts\activate
pip install mediapipe==0.10.9 opencv-python numpy flask
python app.py
```

## Tech Stack
- Python 3.11
- MediaPipe 0.10.9
- OpenCV
- Flask
- NumPy
