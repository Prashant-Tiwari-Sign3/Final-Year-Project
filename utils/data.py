import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, Resize, InterpolationMode, ToImage, ToDtype, Normalize

def create_mask_from_json(json_data, image_size):
    """
    Creates a binary mask based on polygonal regions specified in JSON data.

    Parameters:
    - json_data (list): A list of dictionaries, each containing 'content' key
                       representing a list of points forming a polygon.
                       Each point is a dictionary with 'x' and 'y' keys.
    - image_size (tuple): A tuple representing the size of the target image (height, width).

    Returns:
    - mask (numpy.ndarray): A binary mask with the specified image size,
                           where regions enclosed by the polygons are set to 1,
                           and the rest is set to 0.

    Example:
    ```
    json_data = [{'content': [{'x': 10, 'y': 20}, {'x': 30, 'y': 40}, ...]},
                 {'content': [{'x': 50, 'y': 60}, {'x': 70, 'y': 80}, ...]}]
    image_size = (100, 100)
    mask = create_mask_from_json(json_data, image_size)
    ```
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    for polygon_data in json_data:
        polygon_coordinates = polygon_data['content']
        x_coords = [point['x'] for point in polygon_coordinates]
        y_coords = [point['y'] for point in polygon_coordinates]

        polygon_array = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
        polygon_array = polygon_array.reshape((-1, 1, 2))

        cv.fillPoly(mask, [polygon_array], color=1)

    return mask

SegmentTransform = Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True),
    Resize((520,520), interpolation=InterpolationMode.BICUBIC, antialias=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SegmentationDataset(Dataset):
    """
    Custom PyTorch dataset for image segmentation tasks.

    Args:
        img_list (list): List of file paths for input images.
        mask_list (list): List of file paths for corresponding masks.
        transform (callable): A function/transform to be applied to the input images.

    Attributes:
        images (list): List of file paths for input images.
        masks (list): List of file paths for corresponding masks.
        transform (callable): A function/transform to be applied to the input images.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the indexed sample, consisting of an input image
            and its corresponding mask after applying specified transformations.

    Example:
        dataset = SegmentationDataset(img_list, mask_list, transform)
        sample = dataset[0]
    """

    def __init__(self, img_list, mask_list, transform):
        """
        Initialize the SegmentationDataset.

        Args:
            img_list (list): List of file paths for input images.
            mask_list (list): List of file paths for corresponding masks.
            transform (callable): A function/transform to be applied to the input images.
        """
        self.images = img_list
        self.masks = mask_list
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Returns the indexed sample, consisting of an input image
        and its corresponding mask after applying specified transformations.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed input image and its mask.
        """
        image = cv.cvtColor(cv.imread(self.images[index]), cv.COLOR_BGR2RGB)
        mask = torch.tensor(cv.imread(self.masks[index], cv.IMREAD_GRAYSCALE))
        mask = mask.unsqueeze(0)
        mask = Resize((520, 520), interpolation=InterpolationMode.BICUBIC, antialias=True)(mask)
        image = self.transform(image)
        return image, mask
