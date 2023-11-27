import torch
import cv2 as cv
from PIL import Image
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

class SemanticSegDataset(Dataset):
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

class InstanceSegDataset(Dataset):
    """
    Custom dataset for instance segmentation tasks.

    Args:
        img_list (list): List of file paths for input images.
        mask_list (list): List of file paths for corresponding masks.

    Attributes:
        images (list): List of input image file paths.
        masks (list): List of mask file paths corresponding to the images.

    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(index): Loads and processes the image and its corresponding mask at the given index.

    Returns:
        tuple: Tuple containing the processed image and a dictionary of target information.
               The target dictionary contains "boxes", "labels", and "masks" for instance segmentation.

    Example:
        dataset = InstanceSegDataset(img_list, mask_list)
        img, target = dataset[0]
    """

    def __init__(self, img_list, mask_list):
        """
        Initializes the InstanceSegDataset with a list of image and mask file paths.

        Args:
            img_list (list): List of file paths for input images.
            mask_list (list): List of file paths for corresponding masks.
        """
        self.images = img_list
        self.masks = mask_list

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Loads and processes the image and its corresponding mask at the given index.

        Args:
            index (int): Index of the image in the dataset.

        Returns:
            tuple: Tuple containing the processed image and a dictionary of target information.
                   The target dictionary contains "boxes", "labels", and "masks" for instance segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        masks = np.zeros((num_objs , mask.shape[0] , mask.shape[1]))
        for i in range(num_objs):
            masks[i][mask == i+1] = True
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin , ymin , xmax , ymax])
        boxes = torch.as_tensor(boxes , dtype = torch.float32)
        labels = torch.ones((num_objs,) , dtype = torch.int64)
        masks = torch.as_tensor(masks , dtype = torch.uint8)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        img = ToImage()(img)
        return ToDtype(torch.float32, scale=True)(img) , target