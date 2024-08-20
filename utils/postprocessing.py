# utils/postprocessing.py
import cv2
import os

def save_segmented_objects(image, predictions, save_path='data/segmented_objects/'):
    os.makedirs(save_path, exist_ok=True)
    for i, mask in enumerate(predictions['masks']):
        mask = mask[0].mul(255).byte().cpu().numpy()
        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        segmented_object = cv2.bitwise_and(image, image, mask=thresh)
        cv2.imwrite(f"{save_path}/object_{i}.png", segmented_object)

# Usage
save_segmented_objects(image, output)
