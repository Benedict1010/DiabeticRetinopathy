import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform():
    return A.Compose([
        A.Resize(300, 300),

        A.HorizontalFlip(p=0.5),

        A.Rotate(
            limit=15,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),

        A.Normalize(),

        ToTensorV2()
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(300, 300),

        A.Normalize(),

        ToTensorV2()
    ])