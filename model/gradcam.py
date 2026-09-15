import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A

from torchvision.models import (
    efficientnet_b3,
    EfficientNet_B3_Weights
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT PREPROCESSING
# ============================================================

from preprocessing.crop import crop_black_border


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

NUM_CLASSES = 5
IMAGE_SIZE = 300

TEST_SPLIT = os.path.join(
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

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "best_efficientnet_b3.pth"
)

REPORTS_DIR = os.path.join(
    PROJECT_ROOT,
    "reports"
)

GRADCAM_DIR = os.path.join(
    REPORTS_DIR,
    "gradcam"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DIABETIC RETINOPATHY - GRAD-CAM ERROR ANALYSIS")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading EfficientNet-B3...")

weights = EfficientNet_B3_Weights.DEFAULT

model = efficientnet_b3(
    weights=weights
)

in_features = model.classifier[1].in_features

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3),
    torch.nn.Linear(
        in_features,
        NUM_CLASSES
    )
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading trained checkpoint...")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# Remove DataParallel prefix if present
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key.replace(
            "module.",
            "",
            1
        )

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# VALIDATION PREPROCESSING
# ============================================================

transform = A.Compose([
    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.Normalize()
])


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

if not os.path.exists(TEST_SPLIT):
    raise FileNotFoundError(
        f"Test split not found:\n{TEST_SPLIT}"
    )


test_df = pd.read_csv(
    TEST_SPLIT
)

print(
    f"Test images: "
    f"{len(test_df)}"
)


# ============================================================
# FIND MODEL PREDICTIONS
# ============================================================

print("\nEvaluating test images...")

predictions = []
probabilities_list = []
actual_labels = []

model.eval()

with torch.no_grad():

    for index in range(
        len(test_df)
    ):

        row = test_df.iloc[index]

        image_id = str(
            row["id_code"]
        )

        actual_class = int(
            row["diagnosis"]
        )

        image_path = os.path.join(
            IMAGE_DIR,
            image_id + ".png"
        )

        image = cv2.imread(
            image_path
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read image:\n"
                f"{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = crop_black_border(
            image
        )

        transformed = transform(
            image=image
        )

        processed_image = (
            transformed["image"]
        )

        tensor = torch.tensor(
            processed_image,
            dtype=torch.float32
        )

        tensor = tensor.permute(
            2,
            0,
            1
        )

        tensor = tensor.unsqueeze(
            0
        )

        tensor = tensor.to(
            DEVICE
        )

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        predictions.append(
            predicted_class
        )

        probabilities_list.append(
            probabilities[0].cpu().numpy()
        )

        actual_labels.append(
            actual_class
        )

        if (index + 1) % 50 == 0:
            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(test_df)}"
            )


# ============================================================
# FIND IMPORTANT ERRORS
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)


error_pairs = [
    (3, 2),   # Severe → Moderate
    (4, 2),   # Proliferative → Moderate
    (1, 2),   # Mild → Moderate
    (2, 1),   # Moderate → Mild
    (4, 3),   # Proliferative → Severe
    (3, 4)    # Severe → Proliferative
]


selected_cases = []


for actual_class, predicted_class in error_pairs:

    matches = []

    for index in range(
        len(test_df)
    ):

        if (
            actual_labels[index]
            == actual_class
            and
            predictions[index]
            == predicted_class
        ):
            matches.append(index)

    print(
        f"\n{CLASS_NAMES[actual_class]}"
        f" → "
        f"{CLASS_NAMES[predicted_class]}"
        f": "
        f"{len(matches)} cases"
    )

    # Select first case from each error type
    if len(matches) > 0:

        selected_index = matches[0]

        selected_cases.append(
            (
                selected_index,
                actual_class,
                predicted_class
            )
        )


# ============================================================
# GENERAL MISCLASSIFICATION FALLBACK
# ============================================================

if len(selected_cases) == 0:

    print(
        "\nNo predefined error pair found."
    )

    for index in range(
        len(test_df)
    ):

        if (
            actual_labels[index]
            != predictions[index]
        ):

            selected_cases.append(
                (
                    index,
                    actual_labels[index],
                    predictions[index]
                )
            )

            break


# ============================================================
# GRAD-CAM STORAGE
# ============================================================

activations = None
gradients = None


def forward_hook(
    module,
    input,
    output
):
    global activations

    activations = output


def backward_hook(
    module,
    grad_input,
    grad_output
):
    global gradients

    gradients = grad_output[0]


# ============================================================
# TARGET LAYER
# ============================================================

target_layer = model.features[-1]

forward_handle = (
    target_layer.register_forward_hook(
        forward_hook
    )
)

backward_handle = (
    target_layer.register_full_backward_hook(
        backward_hook
    )
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

print("\n" + "=" * 70)
print("GENERATING GRAD-CAM FOR IMPORTANT ERRORS")
print("=" * 70)


for case_number, (
    index,
    actual_class,
    predicted_class
) in enumerate(
    selected_cases,
    start=1
):

    row = test_df.iloc[index]

    image_id = str(
        row["id_code"]
    )

    image_path = os.path.join(
        IMAGE_DIR,
        image_id + ".png"
    )

    print(
        f"\nCase {case_number}"
    )

    print(
        f"Image ID: {image_id}"
    )

    print(
        f"Actual: "
        f"{CLASS_NAMES[actual_class]}"
    )

    print(
        f"Predicted: "
        f"{CLASS_NAMES[predicted_class]}"
    )


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_image = image.copy()


    # --------------------------------------------------------
    # Crop black border
    # --------------------------------------------------------

    image = crop_black_border(
        image
    )


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    transformed = transform(
        image=image
    )

    processed_image = (
        transformed["image"]
    )

    input_tensor = torch.tensor(
        processed_image,
        dtype=torch.float32
    )

    input_tensor = input_tensor.permute(
        2,
        0,
        1
    )

    input_tensor = input_tensor.unsqueeze(
        0
    )

    input_tensor = input_tensor.to(
        DEVICE
    )

    input_tensor.requires_grad_(True)


    # --------------------------------------------------------
    # Reset Grad-CAM values
    # --------------------------------------------------------

    activations = None
    gradients = None

    model.zero_grad(
        set_to_none=True
    )


    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    outputs = model(
        input_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence = probabilities[
        0,
        predicted_class
    ].item()


    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    score = outputs[
        0,
        predicted_class
    ]

    score.backward()


    # --------------------------------------------------------
    # Validate hooks
    # --------------------------------------------------------

    if activations is None:
        raise RuntimeError(
            "Grad-CAM activations were not captured."
        )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients were not captured."
        )


    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    activation = activations[0]

    gradient = gradients[0]


    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    weights_cam = gradient.mean(
        dim=(1, 2)
    )


    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    cam = torch.zeros(
        activation.shape[1:],
        dtype=torch.float32,
        device=DEVICE
    )

    for channel in range(
        activation.shape[0]
    ):

        cam += (
            weights_cam[channel]
            * activation[channel]
        )


    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = torch.relu(
        cam
    )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    cam = (
        cam
        .detach()
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    if cam.max() > 0:

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min()
        )

    else:

        cam = np.zeros_like(
            cam
        )


    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    cam = cv2.resize(
        cam,
        (
            original_image.shape[1],
            original_image.shape[0]
        )
    )


    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    filename = (
        f"case_{case_number}_"
        f"{CLASS_NAMES[actual_class].replace(' ', '_')}_"
        f"to_"
        f"{CLASS_NAMES[predicted_class].replace(' ', '_')}_"
        f"{image_id}.png"
    )

    output_path = os.path.join(
        GRADCAM_DIR,
        filename
    )


    plt.figure(
        figsize=(15, 5)
    )


    # Original
    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original_image
    )

    plt.title(
        f"Original\n"
        f"Actual: "
        f"{CLASS_NAMES[actual_class]}"
    )

    plt.axis(
        "off"
    )


    # Heatmap
    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        heatmap
    )

    plt.title(
        "Grad-CAM Heatmap"
    )

    plt.axis(
        "off"
    )


    # Overlay
    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        overlay
    )

    plt.title(
        f"Predicted: "
        f"{CLASS_NAMES[predicted_class]}\n"
        f"Confidence: "
        f"{confidence:.2%}"
    )

    plt.axis(
        "off"
    )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        f"Saved: {output_path}"
    )


# ============================================================
# CLEANUP
# ============================================================

forward_handle.remove()
backward_handle.remove()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GRAD-CAM ERROR ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"Generated "
    f"{len(selected_cases)} "
    f"Grad-CAM visualizations."
)

print(
    f"\nResults directory:\n"
    f"{GRADCAM_DIR}"
)

print("=" * 70)