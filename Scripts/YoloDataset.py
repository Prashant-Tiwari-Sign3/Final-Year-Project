import os
import sys
sys.path.append(os.getcwd())

from utils.data import create_yolo_annotations, create_yolo_annotations_v2

if __name__ == '__main__':
    create_yolo_annotations("Data/JSONs/Multiple Books")
    create_yolo_annotations("Data/JSONs/Single Book Images")
    create_yolo_annotations_v2('Data/JSONs/LabelMe/Multiple Books', 'Data/yolo/Multiple Books')