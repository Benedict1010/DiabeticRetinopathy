import cv2


def resize_image(image, size=(300, 300)):
    """
    Resize an RGB image to the model input size.
    """

    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )