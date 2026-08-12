import torch.nn as nn
from torchvision.models import (
    efficientnet_b3,
    EfficientNet_B3_Weights
)


def create_efficientnet_b3(
    num_classes=5,
    pretrained=True,
    dropout=0.3
):
    """
    Creates an EfficientNet-B3 model for
    Diabetic Retinopathy classification.

    Classes:
        0 -> No DR
        1 -> Mild
        2 -> Moderate
        3 -> Severe
        4 -> Proliferative DR
    """

    # ----------------------------------
    # Load pretrained EfficientNet-B3
    # ----------------------------------

    if pretrained:
        weights = EfficientNet_B3_Weights.DEFAULT
    else:
        weights = None

    model = efficientnet_b3(weights=weights)

    # ----------------------------------
    # Get number of classifier inputs
    # ----------------------------------

    in_features = model.classifier[1].in_features

    # ----------------------------------
    # Replace original ImageNet classifier
    # ----------------------------------

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model