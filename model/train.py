import os
import sys
import pandas as pd
import torch

from torch import nn, optim
from torch.utils.data import DataLoader


# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# --------------------------------------------------
# Import project modules
# --------------------------------------------------

from model.datasets import DRDataset
from model.efficientnet import create_efficientnet_b3

from preprocessing.augment import (
    get_train_transform,
    get_val_transform
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 4
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 0


# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_CSV = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "splits",
    "train_split.csv"
)

VAL_CSV = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "splits",
    "val_split.csv"
)

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train_images"
)

MODEL_SAVE_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "logs"
)


# --------------------------------------------------
# Create required directories
# --------------------------------------------------

os.makedirs(
    MODEL_SAVE_DIR,
    exist_ok=True
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 50)
print("DEVICE:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("=" * 50)


# --------------------------------------------------
# Load CSV files
# --------------------------------------------------

print("Loading dataset splits...")

train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)

print(
    f"Training images: {len(train_df)}"
)

print(
    f"Validation images: {len(val_df)}"
)


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

train_dataset = DRDataset(
    dataframe=train_df,
    image_dir=IMAGE_DIR,
    transform=get_train_transform()
)

val_dataset = DRDataset(
    dataframe=val_df,
    image_dir=IMAGE_DIR,
    transform=get_val_transform()
)


# --------------------------------------------------
# Create DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

print("\nLoading EfficientNet-B3...")

model = create_efficientnet_b3(
    num_classes=5,
    pretrained=True,
    dropout=0.3
)

model = model.to(
    device
)


# --------------------------------------------------
# Loss function
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# --------------------------------------------------
# Learning rate scheduler
# --------------------------------------------------

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# --------------------------------------------------
# Training history
# --------------------------------------------------

history = []


# --------------------------------------------------
# Best validation accuracy
# --------------------------------------------------

best_val_accuracy = 0.0


# ==================================================
# TRAINING LOOP
# ==================================================

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 50)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print("=" * 50)


    # ----------------------------------------------
    # TRAINING
    # ----------------------------------------------

    model.train()

    running_train_loss = 0.0

    train_correct = 0
    train_total = 0


    for batch_index, (
        images,
        labels
    ) in enumerate(train_loader):


        # Move data to GPU/CPU

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        # Clear previous gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(
            images
        )


        # Calculate loss

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        # Statistics

        running_train_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_correct += (
            predictions == labels
        ).sum().item()


        train_total += (
            labels.size(0)
        )


        # Print progress

        if (
            (batch_index + 1) % 20 == 0
        ):

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)} "
                f"| Loss: "
                f"{loss.item():.4f}"
            )


    # Calculate training metrics

    train_loss = (
        running_train_loss
        / train_total
    )

    train_accuracy = (
        train_correct
        / train_total
    )


    # ----------------------------------------------
    # VALIDATION
    # ----------------------------------------------

    model.eval()

    running_val_loss = 0.0

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for (
            images,
            labels
        ) in val_loader:


            # Move data to GPU/CPU

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            # Forward pass

            outputs = model(
                images
            )


            # Validation loss

            loss = criterion(
                outputs,
                labels
            )


            running_val_loss += (
                loss.item()
                * images.size(0)
            )


            # Predictions

            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_correct += (
                predictions == labels
            ).sum().item()


            val_total += (
                labels.size(0)
            )


    # Calculate validation metrics

    val_loss = (
        running_val_loss
        / val_total
    )

    val_accuracy = (
        val_correct
        / val_total
    )


    # ----------------------------------------------
    # Update learning rate
    # ----------------------------------------------

    scheduler.step(
        val_loss
    )


    # ----------------------------------------------
    # Print epoch results
    # ----------------------------------------------

    print("\nEpoch Results")

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    # ----------------------------------------------
    # Save history
    # ----------------------------------------------

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })


    # ----------------------------------------------
    # Save best model
    # ----------------------------------------------

    if (
        val_accuracy
        > best_val_accuracy
    ):

        best_val_accuracy = (
            val_accuracy
        )


        model_path = os.path.join(
            MODEL_SAVE_DIR,
            "best_efficientnet_b3.pth"
        )


        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_accuracy":
                    val_accuracy,

                "val_loss":
                    val_loss
            },

            model_path
        )


        print(
            "\n✓ Best model saved!"
        )

        print(
            f"Path: {model_path}"
        )


    # ----------------------------------------------
    # Save training history
    # ----------------------------------------------

    history_df = pd.DataFrame(
        history
    )


    history_path = os.path.join(
        LOG_DIR,
        "training_history.csv"
    )


    history_df.to_csv(
        history_path,
        index=False
    )


# ==================================================
# TRAINING COMPLETE
# ==================================================

print("\n" + "=" * 50)

print(
    "TRAINING COMPLETE"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Model saved in: "
    f"{MODEL_SAVE_DIR}"
)

print("=" * 50)