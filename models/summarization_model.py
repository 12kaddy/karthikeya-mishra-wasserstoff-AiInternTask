# models/summarization_model.py
from transformers import pipeline

class Summarizer:
    def __init__(self):
        self.summarizer = pipeline("summarization")

    def summarize_attributes(self, text):
        summary = self.summarizer(text, max_length=50, min_length=25, do_sample=False)
        return summary[0]['summary_text']

# Usage
summarizer = Summarizer()
summary = summarizer.summarize_attributes(extracted_text)
