import cv2
import streamlit as st
import threading
import base64
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import config
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")  # Full width layout
if "emotion" not in st.session_state:
        st.session_state["emotion"] = "neutral"
# ---------------- Custom CSS for Proper Alignment ---------------- #
st.markdown("""
    <style>
        .block-container {
            padding: 30px !important;
            margin: 0px !important;
        }
        .webrtc-container {
            position: absolute;
            top: 30px;
            left: 20px;
            width: 200px !important;
            height: 200px !important;
            z-index: 1000;
            padding: 10px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- WebRTC Emotion Detection ---------------- #
class FaceEmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.emotion = "neutral"
        self.emotion_lock = threading.Lock()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]  # Process only the first detected face
            face_crop = img[y:y+h, x:x+w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected_emotion = self.update_emotion(face_crop)
        else :
            detected_emotion = "noFace"
            with self.emotion_lock:
                self.emotion = detected_emotion  # Update emotion state
        cv2.putText(img, f"Detected: {detected_emotion}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame.from_ndarray(img, format="bgr24")

    def update_emotion(self, face_crop):
        try:
            if face_crop.size == 0:
                return "noFace"
            
            # ✅ Ensure correct image size without resizing
            result = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
            
            emotion = result[0].get('dominant_emotion', 'neutral').lower() if result else "neutral"
        except:
            emotion = "noFace"

        with self.emotion_lock:
            self.emotion = emotion if emotion in config.EMOTION_VIDEOS else "neutral"
        return self.emotion

# ---------------- Video Playback Section ---------------- #
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def get_current_video(data):
    current_emotion = data
    video_path = config.EMOTION_VIDEOS.get(current_emotion, config.EMOTION_VIDEOS["neutral"])
    return encode_video(video_path)


# ---------------- Streamlit UI ---------------- #
def main():

    st.title("Emotion-Based Reaction System")
    col1, col2 = st.columns([1, 1])

    # Store last emotion update
    if "last_emotion" not in st.session_state:
        st.session_state["last_emotion"] = "neutral"

    with col1:
        st.subheader("Live Emotion Detection")
        processor = webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=FaceEmotionProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        )

        if processor.video_processor:
            detected_emotion = processor.video_processor.emotion

            # Only update session state if emotion has changed
            if detected_emotion != st.session_state["last_emotion"]:
                st.session_state["emotion"] = detected_emotion
                st.session_state["last_emotion"] = detected_emotion

    with col2:
        st.subheader("Video Reactions")
        video_source = get_current_video(st.session_state["emotion"])
        print(st.session_state['emotion'])
        with open("templates/index.html", "r") as file:
            html_template = file.read()

        html_template = html_template.replace("{{ VIDEO_DATA }}", f'"{video_source}"')
        html_template = html_template.replace("{{ EMOTION }}", st.session_state["emotion"])
        st.components.v1.html(html_template, height=1000)
        st_autorefresh(interval=4000, key="refresh")  # Refresh UI every 1s to update video


if __name__ == "__main__":
    main()