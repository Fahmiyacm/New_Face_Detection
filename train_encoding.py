from deepface import DeepFace
import os
import pickle
from tqdm import tqdm

DATASET_PATH = "dataset/train"
ENCODINGS_PATH = "encodings/known_encodings.pkl"


def create_embeddings():
    known_embeddings = {}

    for person_name in os.listdir(DATASET_PATH):
        person_dir = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"Processing {person_name}...")
        embeddings = []

        for img_file in tqdm(os.listdir(person_dir)):
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_dir, img_file)
                try:
                    # Use ArcFace (very accurate)
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        model_name="ArcFace",
                        detector_backend="retinaface",  # or "mtcnn", "opencv"
                        enforce_detection=True
                    )[0]["embedding"]
                    embeddings.append(embedding)
                except:
                    continue

        if embeddings:
            known_embeddings[person_name] = embeddings
            print(f"  → {len(embeddings)} embeddings for {person_name}")

    os.makedirs("encodings", exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(known_embeddings, f)

    print("✅ Embeddings created successfully!")


if __name__ == "__main__":
    create_embeddings()