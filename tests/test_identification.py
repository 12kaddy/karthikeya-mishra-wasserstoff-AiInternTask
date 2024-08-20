import unittest
from models.identification_model import ObjectIdentification
from PIL import Image

class TestIdentification(unittest.TestCase):
    def setUp(self):
        self.identifier = ObjectIdentification()

    def test_identification_output(self):
        # Test that the identification model returns valid scores
        test_image = Image.open("data/segmented_objects/object_0.png")
        scores = self.identifier.identify_object(test_image)
        
        # Ensure the scores are of the expected format
        self.assertIsInstance(scores, torch.Tensor)
        self.assertGreaterEqual(scores.shape[1], 1)  # At least one score per object

if __name__ == "__main__":
    unittest.main()
