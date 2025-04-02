import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
VIDEOS_DIR = os.path.join(BASE_DIR, "static/videos")  # Path to videos folder

EMOTION_VIDEOS = {
    "neutral": os.path.join(VIDEOS_DIR, "dog_normal.mp4"),
    "happy": os.path.join(VIDEOS_DIR, "dog_fly.mp4"),
    "angry": os.path.join(VIDEOS_DIR, "dog_bark.mp4"),
    "noFace": os.path.join(VIDEOS_DIR, "Hi.mp4"),
}
FRAME_SKIP = 10  # Process every 5th frame