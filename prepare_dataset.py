import os
import shutil
import random

# ================== CORRECT SOURCE PATH ==================
# All your person folders are directly in "dataset"
SOURCE_DATASET = r"C:\Users\fahmi\PycharmProjects\FaceDetectionRecognition\dataset"

# Output folders
TRAIN_DIR = os.path.join(SOURCE_DATASET, "train")
TEST_DIR = os.path.join(SOURCE_DATASET, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

print("Starting dataset split...\n")

processed = 0

for person in os.listdir(SOURCE_DATASET):
    person_path = os.path.join(SOURCE_DATASET, person)

    # Skip if it's not a folder or if it's train/test folder
    if not os.path.isdir(person_path) or person in ["train", "test"]:
        continue

    # Get images
    images = [img for img in os.listdir(person_path)
              if img.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

    if len(images) < 5:
        print(f"⚠️ Skipping {person} (too few images)")
        continue

    random.shuffle(images)

    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Create person folders
    train_person_dir = os.path.join(TRAIN_DIR, person)
    test_person_dir = os.path.join(TEST_DIR, person)
    os.makedirs(train_person_dir, exist_ok=True)
    os.makedirs(test_person_dir, exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy(os.path.join(person_path, img), os.path.join(train_person_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(person_path, img), os.path.join(test_person_dir, img))

    print(f"{person:25} → Train: {len(train_images):2} | Test: {len(test_images):2}")
    processed += 1

print("\n" + "=" * 60)
print(f"✅ Dataset splitting completed! Processed {processed} persons")
print(f"Train folder: {TRAIN_DIR}")
print(f"Test folder : {TEST_DIR}")
print("=" * 60)