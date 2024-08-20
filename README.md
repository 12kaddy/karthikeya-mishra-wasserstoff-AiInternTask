# AI Pipeline for Image Segmentation and Object Analysis

This project implements a comprehensive AI pipeline that processes an input image to segment, identify, and analyze objects within the image. The pipeline performs a series of tasks, including image segmentation, object extraction, identification, text/data extraction, attribute summarization, and data mapping. The final output includes a visual representation of the processed image with annotations and a summary table of the mapped data.

## Project Overview

### Objective
To build an end-to-end AI pipeline using deep learning and transformer-based models for segmenting, identifying, and analyzing objects in images. The project integrates various models and tools to achieve the following:
1. **Image Segmentation**: Segment all objects within an input image.
2. **Object Extraction and Storage**: Extract and store segmented objects with unique IDs.
3. **Object Identification**: Identify and describe each object in the segmented images.
4. **Text/Data Extraction**: Extract any text or relevant data from each object.
5. **Summarize Object Attributes**: Summarize the nature and attributes of each object.
6. **Data Mapping**: Map the extracted data and attributes to each object and the master image.
7. **Output Generation**: Generate the final output image with annotations and a summary table.

### Key Features
- **Modular Design**: The project is organized into distinct modules for each stage of the pipeline (e.g., segmentation, identification, text extraction).
- **Streamlit Interface**: A user-friendly Streamlit application to upload images, visualize segmentation results, and view the final output.
- **Comprehensive Documentation**: Detailed setup instructions, usage guidelines, and a README to get started quickly.
- **Test Coverage**: Includes unit tests for all major components to ensure the robustness of the pipeline.
- **Flexible and Extensible**: The pipeline is designed to be flexible, allowing for easy integration of new models or additional features.

## Project Structure

project_root/
* **data/**
  * **input_images/** - Directory for input images
  * **segmented_objects/** - Directory to save segmented object images
  * **output/** - Directory for output images and tables
* **models/**
  * **segmentation_model.py** - Script for segmentation model
  * **identification_model.py** - Script for object identification model
  * **text_extraction_model.py** - Script for text/data extraction model
  * **summarization_model.py** - Script for summarization model
* **utils/**
  * **preprocessing.py** - Script for preprocessing functions
  * **postprocessing.py** - Script for postprocessing functions
  * **data_mapping.py** - Script for data mapping functions
  * **visualization.py** - Script for visualization functions
* **streamlit_app/**
  * **app.py** - Main Streamlit application script
  * **components/** - Directory for Streamlit components
* **tests/**
  * **test_segmentation.py** - Tests for segmentation
  * **test_identification.py** - Tests for identification
  * **test_text_extraction.py** - Tests for text extraction
  * **test_summarization.py** - Tests for summarization
* **README.md** - Project overview and setup instructions
* **requirements.txt** - Required Python packages
* **presentation.pptx** - Presentation slides summarizing the project


## Setup and Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/repository-name.git
    cd repository-name
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit application:
    ```bash
    streamlit run streamlit_app/app.py
    ```

## How to Use

1. Upload an input image through the Streamlit interface.
2. The pipeline will process the image and display the following:
    - Segmented objects with annotations.
    - Detailed information about each object (ID, description, extracted text, attributes).
    - A final output image and a table summarizing all extracted data.
3. Review each step of the pipeline and interact with the output.

## Testing

Run the unit tests to ensure everything works as expected:
```bash
python -m unittest discover -s tests
