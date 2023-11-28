"""
SBStageTraining.py

This script performs training for a single-stage instance segmentation model on a dataset of single book images. It uses a custom training loop implemented in the 'utils.train.TrainLoopV2' module.

Usage:
1. Ensure that the 'utils' module with the required functions and classes is in the 'C:\\College\\Projects\\Final-Year-Project' directory.
2. Make sure that the training and validation sets are available in the following paths:
   - Training images: 'Data/Splits/Training/sb_images.npy'
   - Training masks: 'Data/Splits/Training/sb_masks.npy'
   - Validation images: 'Data/Splits/Validation/sb_images.npy'
   - Validation masks: 'Data/Splits/Validation/sb_masks.npy'
3. Run this script to train the instance segmentation model.
   - The trained model is saved as 'Models/Stage1.pth'.

Note:
- Adjust parameters such as learning rate, batch size, and training duration as needed.
- Ensure that the instance segmentation model is defined in 'utils.models.segmentation.get_instance_segmentation_model'.

Dependencies:
- sys
- torch
- numpy as np
- DataLoader from torch.utils.data
- get_instance_segmentation_model from utils.models.segmentation
- TrainLoopV2 from utils.train
- InstanceSegDataset, collate_fn from utils.data

Author:
    Prashant Tiwari

Date:
    29-11-2023
"""
import sys
sys.path.append("C:\College\Projects\Final-Year-Project")
from utils.models.segmentation import get_instance_segmentation_model
from utils.train import TrainLoopV2
from utils.data import InstanceSegDataset, collate_fn

import torch
import numpy as np
from torch.utils.data import DataLoader

def SBStageTraining():
    train_img = list(np.load("Data/Splits/Training/sb_images.npy"))
    train_masks = list(np.load("Data/Splits/Training/sb_masks.npy"))
    val_img = list(np.load("Data/Splits/Validation/sb_images.npy"))
    val_masks = list(np.load("Data/Splits/Validation/sb_masks.npy"))
    
    train_set = InstanceSegDataset(train_img, train_masks, True)
    val_set = InstanceSegDataset(val_img, val_masks)

    train_loader = DataLoader(train_set, 4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, 4, shuffle=True, collate_fn=collate_fn)

    model = get_instance_segmentation_model(2)
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, 10)

    TrainLoopV2(model, optimizer, train_loader, val_loader, scheduler, 50, 10, batch_loss=5)
    torch.save(model.state_dict(), 'Models/Stage1.pth')

if __name__ == '__main__':
    SBStageTraining()