import unittest
from models.summarization_model import Summarizer

class TestSummarization(unittest.TestCase):
    def setUp(self):
        self.summarizer = Summarizer()

    def test_summarization_output(self):
        # Test that the summarization model returns a string of appropriate length
        sample_text = "This is a test paragraph containing relevant data that needs to be summarized into concise points."
        summary = self.summarizer.summarize_attributes(sample_text)
        
        self.assertIsInstance(summary, str)
        self.assertGreaterEqual(len(summary), 25)  # Check if the summary has at least 25 characters

if __name__ == "__main__":
    unittest.main()
