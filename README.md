# INVENTORY MANAGEMENT WITH IMAGE SEGMENTATION AND OCR

## Overview

This project uses a combination of image segmentation using YOLOv8 and object character recognition with convolutional recurrent neural network to detect the names and number of books in a bookshelf from its image.ok

## Table of Contents


+ [Overview](#overview)
+ [Table of Contents](#table-of-contents)
+ [Dataset](#dataset)
+ [Model Architecture](#model-architecture)
+ [Preprocessing](#preprocessing)
+ [Training](#training)
+ [Evaluation](#evaluation)
+ [Usage](#usage)
+ [Dependencies](#dependencies)

## Dataset

The dataset has been specially created by us for the purpose of this project. It contains ____ images that are separated into three categories:<br>
1. Images with a single book
2. Images with a multiple copies of the same book
3. Images with multiple copies of multiple books

All images are 12 Megapixels(3000x4000 or 4032x2268) in resolution and were preprocessed before being used for training. There are total of **763** single book images for initial phase of model.
