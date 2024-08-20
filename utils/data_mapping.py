# utils/data_mapping.py
import json

def map_data(segments, descriptions, texts, summaries):
    data = []
    for i in range(len(segments)):
        data.append({
            "object_id": i,
            "description": descriptions[i],
            "extracted_text": texts[i],
            "summary": summaries[i]
        })
    return json.dumps(data, indent=4)

# Usage
mapping = map_data(segmentations, object_descriptions, extracted_texts, summaries)
