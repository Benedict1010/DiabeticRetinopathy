import os
import sys

import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------
# Make project root available for imports
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from model.datasets import DRDataset
from model.efficientnet import create_efficientnet_b3

from preprocessing.augment import get_val_transform


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TEST_CSV = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "splits",
    "test_split.csv"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train_images"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "best_efficientnet_b3.pth"
)

REPORT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports"
)

CONFUSION_MATRIX_PATH = os.path.join(
    REPORT_DIR,
    "test_confusion_matrix.png"
)

REPORT_PATH = os.path.join(
    REPORT_DIR,
    "test_evaluation_report.txt"
)


# ---------------------------------------------------------
# Class names
# ---------------------------------------------------------

class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DIABETIC RETINOPATHY MODEL EVALUATION")
print("=" * 60)

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("=" * 60)


# ---------------------------------------------------------
# Check required files
# ---------------------------------------------------------

if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_CSV}"
    )

if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_DIR}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}"
    )


# ---------------------------------------------------------
# Load test dataframe
# ---------------------------------------------------------

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_CSV)

print(f"Test images: {len(test_df)}")

print("\nTest class distribution:")

print(
    test_df["diagnosis"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# Create test dataset
# ---------------------------------------------------------

test_dataset = DRDataset(
    dataframe=test_df,
    image_dir=IMAGE_DIR,
    transform=get_val_transform()
)


# ---------------------------------------------------------
# Create test DataLoader
# ---------------------------------------------------------

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print(f"\nTest batches: {len(test_loader)}")


# ---------------------------------------------------------
# Create model
# ---------------------------------------------------------

print("\nLoading EfficientNet-B3...")

# IMPORTANT:
# pretrained=False because we are loading our already-trained
# model checkpoint. We do not need to download ImageNet weights.

model = create_efficientnet_b3(
    num_classes=5,
    pretrained=False,
    dropout=0.3
)


# ---------------------------------------------------------
# Load trained checkpoint
# ---------------------------------------------------------

print("Loading trained checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# ---------------------------------------------------------
# Handle different checkpoint formats
# ---------------------------------------------------------

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        # The checkpoint itself may simply be a state_dict.
        state_dict = checkpoint

else:
    state_dict = checkpoint


model.load_state_dict(state_dict)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RUNNING TEST EVALUATION")
print("=" * 60)

all_predictions = []
all_labels = []


with torch.no_grad():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        print(
            f"\rProcessed batch "
            f"{batch_index + 1}/{len(test_loader)}",
            end=""
        )


print("\n")


# ---------------------------------------------------------
# Calculate accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3, 4],
    target_names=class_names,
    digits=4,
    zero_division=0
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3, 4]
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(cm)


# ---------------------------------------------------------
# Create reports directory
# ---------------------------------------------------------

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Save confusion matrix image
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

ax.set_title(
    "EfficientNet-B3 - Test Set Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Save text evaluation report
# ---------------------------------------------------------

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DIABETIC RETINOPATHY MODEL EVALUATION REPORT\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Model: EfficientNet-B3\n"
    )

    file.write(
        f"Test Dataset: {TEST_CSV}\n"
    )

    file.write(
        f"Number of Test Images: {len(test_df)}\n"
    )

    file.write(
        f"Device: {device}\n\n"
    )

    file.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report:\n"
    )

    file.write(
        report
    )

    file.write(
        "\n\nConfusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

    file.write("\n")


# ---------------------------------------------------------
# Final message
# ---------------------------------------------------------

print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"\nConfusion matrix saved to:\n"
    f"{CONFUSION_MATRIX_PATH}"
)

print(
    f"\nEvaluation report saved to:\n"
    f"{REPORT_PATH}"
)

print("\nDone.")