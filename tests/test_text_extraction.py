import unittest
from models.text_extraction_model import TextExtraction

class TestTextExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = TextExtraction()

    def test_text_extraction(self):
        # Test that text extraction returns a non-empty string
        test_image_path = "data/segmented_objects/object_0.png"
        extracted_text = self.extractor.extract_text(test_image_path)
        
        self.assertIsInstance(extracted_text, str)
        self.assertGreater(len(extracted_text), 0)  # Ensure some text was extracted

if __name__ == "__main__":
    unittest.main()
