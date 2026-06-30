from deepface import DeepFace
import os
import pickle
from tqdm import tqdm

DATASET_PATH = "dataset/train"
ENCODINGS_PATH = "encodings/encodings.pkl"


def create_embeddings():
    known_embeddings = {}

    for person_name in os.listdir(DATASET_PATH):
        person_dir = os.path.join(DATASET_PATH, person_name)

        if not os.path.isdir(person_dir):
            continue

        print(f"\nProcessing: {person_name}")

        embeddings = []

        for img_file in tqdm(os.listdir(person_dir)):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(person_dir, img_file)

            try:
                # ✅ USE SAME MODEL AS APP (IMPORTANT FIX)
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False
                )[0]["embedding"]

                embeddings.append(embedding)

            except Exception as e:
                print(f"Skipping {img_file} -> {e}")
                continue

        if len(embeddings) > 0:
            known_embeddings[person_name] = embeddings
            print(f"✔ {person_name}: {len(embeddings)} embeddings saved")

    # create folder if not exists
    os.makedirs("encodings", exist_ok=True)

    # save embeddings
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(known_embeddings, f)

    print("\n✅ Embeddings created successfully!")


if __name__ == "__main__":
    create_embeddings()