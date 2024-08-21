# models/identification_model.py
# import torch
# from transformers import CLIPProcessor, CLIPModel

# class ObjectIdentification:
#     def __init__(self):
#         self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     def identify_object(self, image):
#         inputs = self.processor(images=image, return_tensors="pt")
#         outputs = self.model(**inputs)
#         return outputs.logits_per_image  # Returns identification scores

# # Usage
# identifier = ObjectIdentification()
# object_description = identifier.identify_object(image)

# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image

# class ObjectIdentifier:
#     def __init__(self):
#         self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     def identify_object(self, image_path):
#         # Load and process the image
#         image = Image.open(image_path)
#         inputs = self.processor(images=image, return_tensors="pt", padding=True)
        
#         # Perform inference
#         outputs = self.model(**inputs)
#         logits_per_image = outputs.logits_per_image
#         probs = logits_per_image.softmax(dim=1)  # Get probabilities for each label
#         return probs

# # Example usage
# if __name__ == "__main__":
#     identifier = ObjectIdentifier()
#     image_path = "D:/Wasserstoff/karthikeya-mishra-wasserstoff-AiInternTask/data/input_images/2008_000018.jpg"  # Replace with your image path
#     object_description = identifier.identify_object(image_path)
#     print(object_description)

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class ObjectIdentification:
    def __init__(self):
        # Load the pre-trained CLIP model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def identify_object(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        # Perform the model inference
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        # Process the output (Here we're returning raw features for demonstration)
        return outputs

if __name__ == "__main__":
    identifier = ObjectIdentification()
    output = identifier.identify_object("D:/Wasserstoff/karthikeya-mishra-wasserstoff-AiInternTask/data/input_images/2008_000018.jpg")
    print("Identification Output:", output)
