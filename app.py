import cv2
import streamlit as st
import threading
import base64
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import config
from streamlit_autorefresh import st_autorefresh
import json
import os
# ✅ Use a simple shared dictionary at the top level (not a Manager dict)
EMOTION_FILE = "emotion_state.json"

st.set_page_config(layout="wide")

def write_emotion_to_file(emotion):
    with open(EMOTION_FILE, "w") as f:
        json.dump({"emotion": emotion}, f)

def read_emotion_from_file():
    if not os.path.exists(EMOTION_FILE):
        return "neutral"
    with open(EMOTION_FILE, "r") as f:
        try:
            data = json.load(f)
            return data.get("emotion", "neutral")
        except json.JSONDecodeError:
            return "neutral"

# ---------------- CSS ---------------- #
st.markdown("""
    <style>
        .block-container { padding: 30px !important; margin: 0px !important; }
        .webrtc-container {
            position: absolute; top: 30px; left: 20px;
            width: 200px !important; height: 200px !important;
            z-index: 1000; padding: 10px; border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)


class FaceEmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.emotion = "neutral"
        self.emotion_lock = threading.Lock()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def recv(self, frame):
        global shared_data
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_crop = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected_emotion = self.update_emotion(face_crop)
        else:
            detected_emotion = "noFace"
            with self.emotion_lock:
                self.emotion = detected_emotion
                write_emotion_to_file(detected_emotion)  # or detected_emotion depending on context
                print("self",detected_emotion)

        cv2.putText(img, f"Detected: {detected_emotion}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame.from_ndarray(img, format="bgr24")

    def update_emotion(self, face_crop):
        try:
            if face_crop.size == 0:
                return "noFace"

            result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            emotion = result[0].get('dominant_emotion', 'neutral').lower() if result else "neutral"
        except:
            emotion = "noFace"

        with self.emotion_lock:
            self.emotion = emotion if emotion in config.EMOTION_VIDEOS else "neutral"
            write_emotion_to_file(self.emotion)  # or detected_emotion depending on context
 
        return self.emotion


def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def get_current_video(emotion):
    video_path = config.EMOTION_VIDEOS.get(emotion, config.EMOTION_VIDEOS["neutral"])
    return encode_video(video_path)


def main():
    st.title("Emotion-Based Reaction System")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Live Emotion Detection")

        with st.container():
            st.markdown('<div class="webrtc-container">', unsafe_allow_html=True)
            webrtc_streamer(
                key="emotion-detection",
                video_processor_factory=FaceEmotionProcessor,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            )
            st.markdown('</div>', unsafe_allow_html=True)


    with col2:
        st.subheader("Video Reactions")
        current_emotion = read_emotion_from_file()
        video_source = get_current_video(current_emotion)
        with open("templates/index.html", "r") as file:
            html_template = file.read()

        html_template = html_template.replace("{{ VIDEO_DATA }}", f'"{video_source}"')
        html_template = html_template.replace("{{ EMOTION }}", current_emotion)
        st.components.v1.html(html_template, height=1000)
        st_autorefresh(interval=1000, key="refresh")


if __name__ == "__main__":
    main()
