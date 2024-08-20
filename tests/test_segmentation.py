import unittest
import torch
from models.segmentation_model import ImageSegmentation

class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.segmentation = ImageSegmentation()

    def test_segmentation_output(self):
        # Test that the segmentation model returns the expected output format
        test_image_path = "data/input_images/sample.jpg"
        prediction = self.segmentation.segment_image(test_image_path)
        
        # Check that the output is a list containing dictionaries with masks, boxes, and labels
        self.assertIsInstance(prediction, list)
        self.assertIsInstance(prediction[0], dict)
        self.assertIn('masks', prediction[0])
        self.assertIn('boxes', prediction[0])
        self.assertIn('labels', prediction[0])

    def test_mask_shape(self):
        # Ensure that the output mask shape is consistent
        test_image_path = "data/input_images/sample.jpg"
        prediction = self.segmentation.segment_image(test_image_path)
        
        mask = prediction[0]['masks'][0]  # Accessing the first mask
        self.assertEqual(mask.shape[1:], (800, 800))  # Assuming an 800x800 image

if __name__ == "__main__":
    unittest.main()
