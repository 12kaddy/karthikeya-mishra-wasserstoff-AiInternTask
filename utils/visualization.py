# utils/visualization.py
import matplotlib.pyplot as plt
import pandas as pd

def generate_summary_table(data):
    df = pd.DataFrame(data)
    df.to_csv('data/output/summary_table.csv', index=False)
    return df

# Visualization of output image (optional)
def visualize_output(image_path, segments):
    plt.imshow(plt.imread(image_path))
    for segment in segments:
        plt.plot(segment)
    plt.show()

# Usage
summary_table = generate_summary_table(mapped_data)
