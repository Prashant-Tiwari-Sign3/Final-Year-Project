import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50, DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet50_Weights

def get_semantic_segmentation_model(backbone: str = 'mobilenet') -> torch.nn.Module:
    """
    Get an instance of a segmentation model based on the specified backbone.

    Parameters:
        - backbone (str): The backbone architecture for the segmentation model.
          Should be one of ['mobilenet', 'resnet50'].

    Returns:
        - torch.nn.Module: An instance of the segmentation model with the specified backbone.

    Raises:
        - ValueError: If the specified backbone is not one of the valid values.

    Example:
        segmentation_model = get_segmentation_model(backbone='resnet50')
    """
    valid_backbones = ['mobilenet', 'resnet50']
    if backbone not in valid_backbones:
        raise ValueError("Invalid input for backbone. Available backbones are {}".format(valid_backbones))

    if backbone == 'mobilenet':
        SegmentationModel = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    else:
        SegmentationModel = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    SegmentationModel.classifier[-1] = torch.nn.Conv2d(256, 2, kernel_size=1, stride=1)
    return SegmentationModel

def get_instance_segmentation_model(num_classes):
    """
    Returns an instance segmentation model based on Mask R-CNN with a modified head for custom number of classes.

    Parameters:
    - num_classes (int): The number of classes in the target dataset.

    Returns:
    torchvision.models.detection.maskrcnn_resnet50_fpn_v2: A Mask R-CNN model with the specified number of output classes.
    The model's classifier and mask predictor heads are replaced with custom heads to match the target dataset.

    Example:
    ```
    num_classes = 3
    model = get_model_instance_segmentation(num_classes)
    ```
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features           # get number of input features for the classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # replace the pre-trained head with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels    # now get the number of input features for the mask classifier
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )

    return model