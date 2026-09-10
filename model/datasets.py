import os
import cv2
import torch
from torch.utils.data import Dataset

# Our custom retinal image preprocessing
from preprocessing.crop import crop_black_border


class DRDataset(Dataset):
    """
    PyTorch Dataset for the APTOS 2019 Diabetic Retinopathy dataset.

    Each item returned by this dataset contains:
        image -> preprocessed retinal fundus image
        label -> DR severity class

    Labels:
        0 = No DR
        1 = Mild
        2 = Moderate
        3 = Severe
        4 = Proliferative DR
    """

    def __init__(self, dataframe, image_dir, transform=None):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame
            Must contain:
                id_code
                diagnosis

        image_dir : str
            Path to folder containing retinal images.

        transform : Albumentations transform, optional
            Image transformations such as resizing,
            augmentation, normalization and conversion
            to PyTorch tensors.
        """

        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform


    def __len__(self):
        """
        Returns the number of images in the dataset.
        """

        return len(self.dataframe)


    def __getitem__(self, index):
        """
        Loads and preprocesses one image.
        """

        # -----------------------------------
        # 1. Get corresponding CSV row
        # -----------------------------------

        row = self.dataframe.iloc[index]

        image_id = str(row["id_code"])
        label = int(row["diagnosis"])


        # -----------------------------------
        # 2. Construct image path
        # -----------------------------------

        image_path = os.path.join(
            self.image_dir,
            image_id + ".png"
        )


        # -----------------------------------
        # 3. Load retinal image
        # -----------------------------------

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not read retinal image: {image_path}"
            )


        # -----------------------------------
        # 4. Convert OpenCV BGR → RGB
        # -----------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # -----------------------------------
        # 5. Remove black retinal borders
        # -----------------------------------

        image = crop_black_border(image)


        # -----------------------------------
        # 6. Apply transformations
        # -----------------------------------

        if self.transform is not None:

            transformed = self.transform(
                image=image
            )

            image = transformed["image"]


        # -----------------------------------
        # 7. Convert label to PyTorch tensor
        # -----------------------------------

        label = torch.tensor(
            label,
            dtype=torch.long
        )


        # -----------------------------------
        # 8. Return image + diagnosis
        # -----------------------------------

        return image, label