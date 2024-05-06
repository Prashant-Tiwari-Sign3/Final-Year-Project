import os
import json
import torch
import cv2 as cv
import numpy as np
from PIL import Image
from torchvision import tv_tensors
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.ops.boxes import masks_to_boxes

def create_mask_from_json(json_data, image_size):
    """
    Creates a binary mask based on polygonal regions specified in JSON data.

    Parameters:
    - json_data (list): A list of dictionaries, each containing 'content' key representing a list of points forming a polygon. Each point is a dictionary with 'x' and 'y' keys.
    - image_size (tuple): A tuple representing the size of the target image (height, width).

    Returns:
    - mask (numpy.ndarray): A binary mask with the specified image size, where regions enclosed by the polygons are set to 1, and the rest is set to 0.

    Example:
    ```
    json_data = [{'content': [{'x': 10, 'y': 20}, {'x': 30, 'y': 40}, ...]},
                {'content': [{'x': 50, 'y': 60}, {'x': 70, 'y': 80}, ...]}]
    image_size = (100, 100)
    mask = create_mask_from_json(json_data, image_size)
    ```
    """
    mask = np.zeros(image_size, dtype=np.uint8)
    for i, polygon_data in enumerate(json_data):
        polygon_coordinates = polygon_data['content']
        x_coords = [point['x'] for point in polygon_coordinates]
        y_coords = [point['y'] for point in polygon_coordinates]

        polygon_array = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
        polygon_array = polygon_array.reshape((-1, 1, 2))

        cv.fillPoly(mask, [polygon_array], color=i+1)

    return mask

SegmentTransform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((520,520), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        mask = v2.Resize((520, 520), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)(mask)
        image = self.transform(image)
        return image, mask

class InstanceSegDataset(Dataset):
    """
    Custom dataset class for instance segmentation tasks.

    Args:
        img_list (list): List of file paths to input images.
        mask_list (list): List of file paths to corresponding mask images.
        train (bool, optional): Flag indicating whether the dataset is used for training.
            Defaults to False.

    Attributes:
        images (list): List of file paths to input images.
        masks (list): List of file paths to corresponding mask images.
        transform (torchvision.transforms.Compose): Composition of data transformations.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(index): Loads and preprocesses the image and mask at the specified index.

    Example:
        img_list = ["path/to/img1.jpg", "path/to/img2.jpg", ...]
        mask_list = ["path/to/mask1.jpg", "path/to/mask2.jpg", ...]
        dataset = InstanceSegDatasetv2(img_list, mask_list, train=True)
    """

    def __init__(self, img_list, mask_list, train: bool = False):
        """
        Initializes the InstanceSegDatasetv2 dataset.

        Args:
            img_list (list): List of file paths to input images.
            mask_list (list): List of file paths to corresponding mask images.
            train (bool, optional): Flag indicating whether the dataset is used for training.
                Defaults to False.
        """
        self.images = img_list
        self.masks = mask_list
        transforms = []
        if train:
            transforms.append(v2.RandomHorizontalFlip(0.55))
            transforms.append(v2.RandomVerticalFlip(0.55))
            transforms.append(v2.RandomRotation(50))
        transforms.append(v2.ToDtype(torch.float, scale=True))
        transforms.append(v2.ToPureTensor())
        self.transform = v2.Compose(transforms)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Loads and preprocesses the image and mask at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the preprocessed image and a dictionary of target information.
        """
        img = read_image(self.images[index])
        mask = read_image(self.masks[index])

        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        boxes = masks_to_boxes(masks)
        labels = torch.ones((num_objs,), dtype=torch.int8)
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=v2.functional.get_size(img))
        target["labels"] = labels
        target["masks"] = tv_tensors.Mask(masks)

        img, target = self.transform(img, target)
        return img, target

def collate_fn(batch):
    """
    collate_fn: Function to be used as the collate_fn parameter in PyTorch DataLoader.
    This function is used to customize the behavior of collating samples in a batch.

    Parameters:
        - batch (list): List of individual samples, each typically containing input features and target labels.

    Returns:
        - tuple: A tuple containing batches of each element in the input samples. The elements are grouped based on their position in the original samples.
    """
    return tuple(zip(*batch))


def create_yolo_annotations(json_path: str):
    json_list = os.listdir(json_path)
    yolo_dataset_path = json_path.replace('JSONs', 'yolo')
    if not os.path.exists(yolo_dataset_path):
        os.makedirs(yolo_dataset_path)
    yolo_list = [yolo_dataset_path + '/' + file.replace('json', 'txt') for file in json_list]
    json_list = [json_path+'/'+file for file in json_list]
    image_list = [file.replace('json', 'jpg') for file in json_list]
    image_list = [file.replace('JSONs', 'Images') for file in image_list]

    for i in range(len(json_list)):
        with open(json_list[i], 'r') as f:
            data = json.load(f)
        img = Image.open(image_list[i])
        img_width = img.size[0]
        img_height = img.size[1]
        with open(yolo_list[i], 'w') as out_file:
            for obj in data:
                polygon = obj['content']
                segmentation = []
                for point in polygon:
                    x, y = point.values()
                    normalized_x = x / img_width
                    normalized_y = y / img_height
                    segmentation.append(f"{normalized_x} {normalized_y}")

                segmentation_str = " ".join(segmentation)
                class_index = 0  
                out_file.write(f"{class_index} {segmentation_str}\n")

def create_yolo_annotations_v2(json_path: str, destination_path: str):
    json_list = os.listdir(json_path)
    txt_list = [file.replace('json', 'txt') for file in json_list]
    json_list = [json_path + '/' + file for file in json_list]
    txt_list = [destination_path + '/' + file for file in txt_list]

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for path1, path2 in zip(json_list, txt_list):
        with open(path1, 'r') as a:
            data = json.load(a)

        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        with open(path2, "w") as file:
            for shape in data["shapes"]:
                points = shape["points"]
                normalized_points = [(x / image_width, y / image_height) for x, y in points]
                line = f"{0} " + " ".join(f"{x:.8f} {y:.8f}" for x, y in normalized_points)
                file.write(line + "\n")