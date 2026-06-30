import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
import os
import tempfile
from deepface import DeepFace
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Face Recognition System", layout="wide")
st.title("🔍 Face Detection & Recognition (Render Optimized)")

# Reduce TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------- LOAD ENCODINGS ----------------
@st.cache_resource
def load_encodings():
    with open("encodings/encodings.pkl", "rb") as f:
        return pickle.load(f)

known_encodings = load_encodings()

# ---------------- SIDEBAR ----------------
option = st.sidebar.selectbox(
    "Choose Mode",
    ["Image Upload", "Model Metrics"]
)

# ---------------- IMAGE UPLOAD ----------------
if option == "Image Upload":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)

        # SAFE TEMP FILE
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name
            image.save(temp_path)

        try:
            # Detect faces (lightweight)
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend="opencv",
                enforce_detection=False
            )

            if len(faces) == 0:
                st.warning("No face detected")
            else:
                st.success(f"{len(faces)} face(s) detected")

                # Load image
                img = cv2.imread(temp_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # COMPUTE EMBEDDING ONLY ONCE (IMPORTANT FIX)
                embedding_obj = DeepFace.represent(
                    img_path=temp_path,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False
                )[0]

                embedding = np.array(embedding_obj["embedding"])

                best_name = "Unknown"
                min_distance = float("inf")

                # Compare with known embeddings
                for name, emb_list in known_encodings.items():
                    for known_emb in emb_list:
                        known_emb = np.array(known_emb)

                        distance = 1 - np.dot(embedding, known_emb) / (
                            np.linalg.norm(embedding) * np.linalg.norm(known_emb)
                        )

                        if distance < min_distance:
                            min_distance = distance
                            best_name = name

                confidence = max(0, (1 - min_distance) * 100)
                # ---------------- DRAW IMAGE ----------------
                for face in faces:
                    x = face["facial_area"]["x"]
                    y = face["facial_area"]["y"]
                    w = face["facial_area"]["w"]
                    h = face["facial_area"]["h"]

                    color = (0, 255, 0) if min_distance < 0.5 else (255, 0, 0)
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                # Convert for Streamlit display
                st.image(img, caption="Detection Result", width=500)

                # ---------------- RESULT BELOW IMAGE ----------------
                st.markdown("## 🎯 Recognition Result")

                if min_distance < 0.5:
                    st.success(f"👤 Name: **{best_name}**")
                else:
                    st.warning("👤 Name: **Unknown**")

                st.info(f"📊 Confidence Score: **{confidence:.2f}%**")



        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# ---------------- METRICS ----------------
elif option == "Model Metrics":
    st.subheader("📊 Model Evaluation on Test Set")

    results_file = "evaluation_results1.pkl"

    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)

        acc = results["accuracy"]
        st.success(f"**Overall Accuracy: {acc * 100:.2f}%**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precision", f"{results['precision'] * 100:.2f}%")
        with col2:
            st.metric("Recall", f"{results['recall'] * 100:.2f}%")
        with col3:
            st.metric("F1-Score", f"{results['f1'] * 100:.2f}%")

        # Plots
        st.subheader("📈 Accuracy Plots")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Bar Chart
        ax[0].bar(['Accuracy', 'Precision', 'Recall'],
                  [acc, results['precision'], results['recall']])
        ax[0].set_ylim(0, 1)
        ax[0].set_title("Performance Metrics")

        # Simulated Epochs Plot
        epochs = list(range(1, 11))
        sim_acc = [0.65 + i * 0.03 for i in range(10)]
        ax[1].plot(epochs, sim_acc, marker='o', linewidth=2, color='blue')
        ax[1].set_title("Accuracy vs Epochs (Simulated Training)")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].grid(True)

        st.pyplot(fig)
