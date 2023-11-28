"""
SingleBookMasks.py

This script generates binary masks from JSON files containing polygonal annotations for images.
The masks are saved as PNG images in a specified directory.

Usage:
1. Ensure that the 'utils' module with the 'create_mask_from_json' function is in the
   'C:\College\Projects\Final-Year-Project' directory.
2. Place image files in the 'Data/Single Book Images' directory.
3. Place corresponding JSON files in the 'Data/JSONs/Single Book Images' directory.
4. Run this script to generate masks, which will be saved in the 'Data/Masks/Single Book Images' directory.

Note: This script assumes that each image has a corresponding JSON file, and the naming
      convention for JSON files matches the corresponding image file.

Dependencies:
- OpenCV (cv2)
- json
- os
- sys

Author:
    Prashant Tiwari

Date:
    23-11-2023
"""
import sys
sys.path.append("C:\College\Projects\Final-Year-Project")
from utils.data import create_mask_from_json

import os
import json
import cv2 as cv

def CreateMask():
    img_path = "Data/Images/Single Book Images"
    img_list = os.listdir(img_path)

    sizes = []
    for image in img_list:
        path = os.path.join(img_path, image)
        img = cv.imread(path)
        size = img.shape[:2]
        sizes.append(size)

    json_path = "Data/JSONs/Single Book Images"
    json_list = os.listdir(json_path)

    mask_path = "Data/Masks/Single Book Images"

    for i, js in enumerate(json_list):
        path = os.path.join(json_path, js)
        json_data = json.load(open(path))

        mask = create_mask_from_json(json_data, sizes[i])
        path_mask = os.path.join(mask_path, js.replace('.json', '.png'))
        cv.imwrite(path_mask, mask)

if __name__ == '__main__':
    CreateMask()