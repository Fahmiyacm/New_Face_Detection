import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
import os
from deepface import DeepFace
import matplotlib.pyplot as plt
import tempfile

st.set_page_config(page_title="Face Recognition System", layout="wide")
st.title("🔍 End-to-End Face Detection & Recognition System")


# ===================== LOAD ENCODINGS =====================
@st.cache_resource
def load_encodings():
    try:
        with open("encodings/known_encodings.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Encodings not found! Please run `train_encodings.py` first.")
        return {}
    except Exception as e:
        st.error(f"Error loading encodings: {e}")
        return {}


known_encodings = load_encodings()

# ===================== SIDEBAR =====================
st.sidebar.header("Options")
option = st.sidebar.selectbox("Choose Mode",
                              ["Image Upload", "Model Metrics & Plots"])

# ===================== IMAGE UPLOAD =====================
if option == "Image Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Show uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=500)

        # Save temporarily


        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            temp_path = tmp.name
            image.save(temp_path)

        try:
            # Face Detection
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend="mtcnn",
                enforce_detection=False
            )

            if not faces:
                st.error("No faces detected in the image!")
            else:
                st.success(f"✅ Detected {len(faces)} face(s)")

                # Load image for drawing
                img_array = cv2.imread(temp_path)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

                for face_obj in faces:
                    area = face_obj['facial_area']
                    x, y, w, h = area['x'], area['y'], area['w'], area['h']

                    # Get embedding
                    embedding_obj = DeepFace.represent(
                        img_path=temp_path,
                        model_name="ArcFace",
                        detector_backend="mtcnn",
                        enforce_detection=False,
                        align=True
                    )[0]

                    embedding = np.array(embedding_obj["embedding"])

                    # Find best match
                    best_name = "Unknown"
                    min_distance = float("inf")

                    for name, emb_list in known_encodings.items():
                        for known_emb in emb_list:
                            known_emb = np.array(known_emb)
                            distance = 1 - np.dot(embedding, known_emb) / (
                                        np.linalg.norm(embedding) * np.linalg.norm(known_emb))
                            if distance < min_distance:
                                min_distance = distance
                                best_name = name

                    confidence = max(0, (1 - min_distance) * 100)

                    # Draw rectangle only
                    color = (0, 255, 0) if min_distance < 0.50 else (255, 0, 0)
                    cv2.rectangle(img_array, (x, y), (x + w, y + h), color, 3)

                    # Show result as text
                    if min_distance < 0.50:
                        st.success(f"**Recognized Person: {best_name}**")
                        st.info(f"**Confidence Score: {confidence:.2f}%**")
                    else:
                        st.warning(f"**Person: Unknown**")
                        st.info(f"**Closest Match: {best_name}** (Confidence: {confidence:.2f}%)")

                # Show final image
                st.image(img_array, caption="Detection & Recognition Result", width=500)

        except Exception as e:
            st.error(f"Processing Error: {str(e)}")

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ===================== EVALUATION =====================
elif  option == "Model Metrics & Plots":
    st.subheader("📊 Model Evaluation on Test Set")

    results_file = "evaluation_results.pkl"

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

