import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50, DeepLabV3_MobileNet_V3_Large_Weights, DeepLabV3_ResNet50_Weights

def get_segmentation_model(backbone: str = 'mobilenet') -> torch.nn.Module:
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

    SegmentationModel.classifier[-1] = torch.nn.Conv2d(256, 1, kernel_size=1, stride=1)
    return SegmentationModel
