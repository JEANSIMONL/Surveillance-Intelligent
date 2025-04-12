import streamlit as st
import cv2
from ultralytics import YOLO
from playsound3 import playsound
import threading
import time

model_feu = YOLO('/home/wecode-071/Surveillance-Intelligent/projet_predictions_model/feu.pt')
sound_feu = '/home/wecode-071/Surveillance-Intelligent/projet_predictions_model/feu.mp3'

model_fumee = YOLO('/home/wecode-071/Surveillance-Intelligent/projet_predictions_model/smoke.pt')
sound_fumee = '/home/wecode-071/Surveillance-Intelligent/projet_predictions_model/smoke.mp3'

model_accident = YOLO('/home/wecode-071/Surveillance-Intelligent/projet_predictions_model/accident.pt')
sound_accident = '/home/wecode-071/Surveillance-Intelligent/projet_predictions_model/accident.mp3'

st.title("🛑 Système de Surveillance Intelligent")
st.write("Choisis un mode de détection et lance la caméra.")

mode = st.radio("🎯 Mode :", ["🔥 Feu + Fumée", "💥 Accident"])

st.sidebar.title("🔧 Réglages image")
luminosite = st.sidebar.slider("Luminosité", 0, 100, 50)
contraste = st.sidebar.slider("Contraste", 0, 100, 50)

if "cam_active" not in st.session_state:
    st.session_state.cam_active = False

if st.button("📷 Activer / Désactiver la Caméra"):
    st.session_state.cam_active = not st.session_state.cam_active

st.write("📡 Caméra :", "✅ Active" if st.session_state.cam_active else "❌ Off")

dernier_son = {
    "feu": 0,
    "fumee": 0,
    "accident": 0
}
pause_son = 5 

if st.session_state.cam_active:
    cap = cv2.VideoCapture(0)
    vue = st.empty()

    while cap.isOpened():
        ok, image = cap.read()
        if not ok:
            st.error("❌ Caméra non détectée.")
            break

        alpha = contraste / 50  
        beta = luminosite - 50  
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        maintenant = time.time()

        # Mode Feu + Fumée
        if mode == "🔥 Feu + Fumée":
            feu = model_feu(image, conf=0.8)
            for result in feu:
                if result.boxes and maintenant - dernier_son["feu"] > pause_son:
                    st.warning("🔥 Feu détecté !")
                    threading.Thread(target=playsound, args=(sound_feu,), daemon=True).start()
                    dernier_son["feu"] = maintenant
                    break

            fumee = model_fumee(image, conf=0.8)
            for result in fumee:
                if result.boxes and maintenant - dernier_son["fumee"] > pause_son:
                    st.warning("💨 Fumée détectée !")
                    threading.Thread(target=playsound, args=(sound_fumee,), daemon=True).start()
                    dernier_son["fumee"] = maintenant
                    break

        # Mode Accident
        elif mode == "💥 Accident":
            accident = model_accident(image, conf=0.85)
            for result in accident:
                if result.boxes and maintenant - dernier_son["accident"] > pause_son:
                    st.warning("💥 Accident détecté !")
                    threading.Thread(target=playsound, args=(sound_accident,), daemon=True).start()
                    dernier_son["accident"] = maintenant
                    break

        vue.image(image, channels="BGR", use_container_width=True)

    cap.release()

else:
    st.info("🎥 Caméra désactivée. Clique sur le bouton pour la lancer.")
