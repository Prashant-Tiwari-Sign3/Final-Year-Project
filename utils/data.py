import cv2 as cv
import numpy as np

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