# models/text_extraction_model.py
import pytesseract
from PIL import Image

class TextExtraction:
    def __init__(self):
        # Set the path to the Tesseract executable
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def extract_text(self, image_path):
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text

# Usage
extractor = TextExtraction()
extracted_text = extractor.extract_text('data/segmented_objects/object_0.png')
print(extracted_text)
