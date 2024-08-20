# models/segmentation_model.py
import torch
import torchvision
from PIL import Image
import torchvision.transforms as T

class ImageSegmentation:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def segment_image(self, image_path):
        image = Image.open(image_path)
        transform = T.Compose([T.ToTensor()])
        image = transform(image)
        with torch.no_grad():
            prediction = self.model([image])
        return prediction  # Returns segmented regions

# Usage
segmentation = ImageSegmentation()
output = segmentation.segment_image('data/input_images/sample.jpg')
