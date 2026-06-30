import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

TEST_PATH = "dataset/test"
ENCODINGS_PATH = "encodings/encodings.pkl"
RESULTS_PATH = "evaluation_results1.pkl"


def run_evaluation():
    print("📊 Starting Evaluation...\n")

    # Load encodings
    with open(ENCODINGS_PATH, "rb") as f:
        known_encodings = pickle.load(f)

    y_true = []
    y_pred = []

    # Count total images
    total_images = 0
    for person in os.listdir(TEST_PATH):
        person_dir = os.path.join(TEST_PATH, person)
        if os.path.isdir(person_dir):
            total_images += len([
                f for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])

    processed = 0

    for person_name in os.listdir(TEST_PATH):
        person_dir = os.path.join(TEST_PATH, person_name)

        if not os.path.isdir(person_dir):
            continue

        print(f"Evaluating: {person_name}")

        for img_file in tqdm(os.listdir(person_dir)):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(person_dir, img_file)

            try:
                # IMPORTANT: MUST MATCH YOUR encodings.pkl MODEL
                embedding_obj = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet",   # ✔ keep same as encodings
                    detector_backend="opencv",
                    enforce_detection=False
                )

                if len(embedding_obj) == 0:
                    continue

                embedding = np.array(embedding_obj[0]["embedding"])

                best_name = "Unknown"
                min_distance = float("inf")

                for name, emb_list in known_encodings.items():
                    for known_emb in emb_list:
                        known_emb = np.array(known_emb)

                        # cosine distance
                        distance = 1 - np.dot(embedding, known_emb) / (
                            np.linalg.norm(embedding) * np.linalg.norm(known_emb)
                        )

                        if distance < min_distance:
                            min_distance = distance
                            best_name = name

                y_true.append(person_name)

                # threshold check
                if min_distance < 0.5:
                    y_pred.append(best_name)
                else:
                    y_pred.append("Unknown")

            except Exception as e:
                print(f"Skipping {img_file}: {e}")
                continue

            processed += 1
            print(f"Progress: {processed}/{total_images}")

    # ❗ SAFETY CHECK
    if len(y_true) == 0 or len(y_pred) == 0:
        print("❌ No valid predictions found! Check dataset or encodings.")
        return

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "y_pred": y_pred
    }

    # Save results
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)

    print("\n✅ Evaluation Completed!")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")


if __name__ == "__main__":
    run_evaluation()