"""
SplitsSets.py

This script divides a dataset of single book images and their corresponding masks into training, testing, and validation sets using the scikit-learn library.

Usage:
1. Ensure that the 'Data/Images/Single Book Images' directory contains single book images.
2. Ensure that the 'Data/Masks/Single Book Images' directory contains corresponding masks.
3. Run this script to split the dataset and save paths of images and masks for training, testing, and validation.
   - Training set paths are saved in 'Data/Splits/Training/sb_images.npy' and 'Data/Splits/Training/sb_masks.npy'.
   - Testing set paths are saved in 'Data/Splits/Testing/sb_images.npy' and 'Data/Splits/Testing/sb_masks.npy'.
   - Validation set paths are saved in 'Data/Splits/Validation/sb_images.npy' and 'Data/Splits/Validation/sb_masks.npy'.

Note: Adjust the test_size and random_state parameters in the train_test_split function for different splits.

Dependencies:
- os
- numpy as np
- train_test_split from sklearn.model_selection

Author:
    Prashant Tiwari

Date:
    29-11-2023
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split

def SplitsSets():
    sb_img_dir = "Data/Images/Single Book Images"
    sb_mask_dir = "Data/Masks/Single Book Images"
    sb_img_list = os.listdir(sb_img_dir)
    sb_mask_list = os.listdir(sb_mask_dir)

    sb_img_list = [os.path.join(sb_img_dir, img) for img in sb_img_list]
    sb_mask_list = [os.path.join(sb_mask_dir, mask) for mask in sb_mask_list]

    sb_img_list = np.array(sb_img_list)
    sb_mask_list = np.array(sb_mask_list)

    train_img, test_img, train_mask, test_mask = train_test_split(sb_img_list, sb_mask_list, test_size=0.15, shuffle=True, random_state=1)
    train_img, val_img, train_mask, val_mask = train_test_split(train_img, train_mask, test_size=0.12, shuffle=True, random_state=1)

    np.save('Data/Splits/Training/sb_images.npy', train_img)
    np.save('Data/Splits/Training/sb_masks.npy', train_mask)
    np.save('Data/Splits/Testing/sb_images.npy', test_img)
    np.save('Data/Splits/Testing/sb_masks.npy', test_mask)
    np.save('Data/Splits/Validation/sb_images.npy', val_img)
    np.save('Data/Splits/Validation/sb_masks.npy', val_mask)

if __name__ == '__main__':
    SplitsSets()