import cv2
import streamlit as st
import numpy as np
import threading
import time
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import config

# Global Variables
reaction_frame = None
lock = threading.Condition()
video_data = None

# Face Detection Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class FaceEmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.emotion = "neutral"
        self.current_video = "neutral"
        self.force_playing = False
        self.emotion_lock = threading.Lock()
        self.frame_count = 0  # Track frame count to skip frames

        # Start a separate thread for DeepFace analysis
        self.emotion_thread = threading.Thread(target=self.detect_emotion, daemon=True)
        self.emotion_thread.start()

    def recv(self, frame):
        global reaction_frame

        self.frame_count += 1
        if self.frame_count % config.FRAME_SKIP != 0:
            return frame  # Skip frames to optimize CPU usage

        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_crop = img[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            threading.Thread(target=self.update_emotion, args=(face_crop,)).start()
        else:
            self.update_emotion("no_face")

        with self.emotion_lock:
            detected_emotion = self.emotion

        cv2.putText(img, f"Detected: {detected_emotion}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame.from_ndarray(img, format="bgr24")

    def update_emotion(self, face_crop):
        """Runs DeepFace analysis in a separate thread to avoid blocking."""
        try:
            if isinstance(face_crop, str):  # No face detected
                emotion = "no_face"
            else:
                face_crop = cv2.resize(face_crop, (48, 48))  # Speed optimization
                emotion = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)[0]['dominant_emotion'].lower()
        except:
            emotion = "neutral"

        with self.emotion_lock:
            self.emotion = emotion if emotion in config.EMOTION_VIDEOS else "neutral"

    def detect_emotion(self):
        """Continuously checks if video needs switching."""
        while True:
            with self.emotion_lock:
                emotion = self.emotion

            if emotion in config.EMOTION_VIDEOS and emotion != self.current_video:
                self.current_video = emotion
                fetch_video_from_google_drive(emotion)

            time.sleep(0.1)  # Avoid CPU overuse

def fetch_video_from_google_drive(emotion):
    """Fetches video from Google Drive and loads it into a cv2 VideoCapture object."""
    global video_data

    video_url = config.EMOTION_VIDEOS.get(emotion, config.EMOTION_VIDEOS["neutral"])
    
    try:
        video_data = cv2.VideoCapture(video_url)
        if not video_data.isOpened():
            st.error(f"Failed to load video for {emotion}")
    except Exception as e:
        st.error(f"Error fetching video: {e}")

def process_reaction_video():
    """Continuously processes video frames from the streamed data."""
    global reaction_frame, video_data

    while True:
        if video_data is None:
            fetch_video_from_google_drive("neutral")  # Default video
        
        video_cap = video_data  # Use the already initialized VideoCapture object

        while video_cap.isOpened():
            ret, video_frame = video_cap.read()
            if not ret:
                fetch_video_from_google_drive("neutral")  # Loop back to default
                break

            video_frame = cv2.resize(video_frame, (400, 400))
            
            with lock:
                reaction_frame = video_frame.copy()

            time.sleep(0.03)

def main():
    st.title("Emotion-Based Reaction System")
    st.write("Detects emotions and plays corresponding reaction videos.")

    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=FaceEmotionProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    )

    reaction_placeholder = st.empty()

    reaction_thread = threading.Thread(target=process_reaction_video, daemon=True)
    reaction_thread.start()

    while webrtc_ctx and webrtc_ctx.state.playing:
        with lock:
            if reaction_frame is not None:
                reaction_placeholder.image(reaction_frame, channels="BGR", use_container_width=True)
        time.sleep(0.03)

if __name__ == "__main__":
    main()
