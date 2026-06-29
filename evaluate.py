import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

TEST_PATH = "dataset/test"
ENCODINGS_PATH = "encodings/known_encodings.pkl"
RESULTS_PATH = "evaluation_results.pkl"


def run_evaluation():
    print("📊 Starting Evaluation on Test Set...\n")

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
            total_images += len([f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    processed = 0

    for person_name in os.listdir(TEST_PATH):
        person_dir = os.path.join(TEST_PATH, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"Evaluating: {person_name}")

        for img_file in tqdm(os.listdir(person_dir)):
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(person_dir, img_file)

            try:
                # Get embedding
                embedding_obj = DeepFace.represent(
                    img_path=img_path,
                    model_name="ArcFace",
                    detector_backend="mtcnn",  # Fast
                    enforce_detection=False
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

                y_true.append(person_name)
                y_pred.append(best_name if min_distance < 0.50 else "Unknown")

            except Exception:
                continue

            processed += 1
            print(f"  Progress: {processed}/{total_images}")

    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

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
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()