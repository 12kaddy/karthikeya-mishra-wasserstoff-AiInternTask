# models/segmentation_model.py
# import torch
# import torchvision
# from PIL import Image
# import torchvision.transforms as T
# from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# class ImageSegmentation:
#     def __init__(self):
        
#         self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
#         self.model.eval()

#     def segment_image(self, image_path):
#         image = Image.open(image_path)
#         transform = T.Compose([T.ToTensor()])
#         image = transform(image)
#         with torch.no_grad():
#             prediction = self.model([image])
#         return prediction  # Returns segmented regions

# # Usage
# segmentation = ImageSegmentation()
# output = segmentation.segment_image('D:/Wasserstoff/karthikeya-mishra-wasserstoff-AiInternTask/data/input_images/2008_000018.jpg')

# models/segmentation_model.py
import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import os
import numpy as np

class ImageSegmentation:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.model.eval()
        self.output_dir = 'D:/Wasserstoff/karthikeya-mishra-wasserstoff-AiInternTask/data/segmented_objects'
        os.makedirs(self.output_dir, exist_ok=True)

    def segment_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image)
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        # Process and save each segmented object
        for idx, (mask, score) in enumerate(zip(predictions[0]['masks'], predictions[0]['scores'])):
            if score > 0.5:  # Threshold to filter out low-confidence detections
                mask = mask.squeeze().cpu().numpy()
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(os.path.join(self.output_dir, f'object_{idx}.png'))

        return predictions

# Usage
if __name__ == "__main__":
    segmentation = ImageSegmentation()
    # Replace with the path to your input image
    input_image_path = 'data/input_images/2008_000018.jpg'
    segmentation.segment_image(input_image_path)

