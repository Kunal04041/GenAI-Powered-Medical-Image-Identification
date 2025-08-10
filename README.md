# GenAI-Powered-Medical-Image-Identification


## 1.Project Overview 
This project is designed to automatically extract images from **PDF documents** or **web pages** and classify them as **medical** or **non-medical** using the **BiomedCLIP** model from OpenCLIP.  

The system supports a variety of image formats, performs intelligent preprocessing (resizing, format conversion, and optimization), and processes images in batches for efficiency.  
Once classified, the results are displayed visually with confidence scores and also saved in a CSV file for easy analysis.  

This tool can be useful for **medical researchers**, **document processing systems**, or **content filtering applications** where separating medical imagery from general imagery is essential.  

---

##  2.Features  
- **Image Extraction from PDFs**  
  Convert PDF pages into images using `pdf2image` and Poppler utilities.  

- **Image Extraction from Websites**  
  Scrape and download images from any public webpage using `BeautifulSoup` and `requests`.  

- **Medical vs. Non-Medical Classification**  
  Leverage the state-of-the-art **BiomedCLIP** model for accurate image classification.  

- **Batch Image Processing**  
  Handle multiple images at once to speed up classification.  

- **Preprocessing Pipeline**  
  Automatically converts non-RGB images, resizes large images, and optimizes them for inference.  

- **Result Visualization**  
  View classified images in a clean grid layout with color-coded labels and confidence scores.  

- **CSV Export**  
  Save detailed classification results (image name, prediction, confidence) for reporting or further analysis.

## 3.**Setup Instructions**
3.1 Download the `app.py` file and `requirements.txt` file to a folder on your local machine.

3.2 **Create a virtual environment** to keep dependencies isolated:
 


   Run this command in your terminal or command prompt to create a new virtual environment:
   ```python
   python -m venv myenv
   ```
3.3 Install Dependencies

Install all required Python packages by running:
```python
pip install -r requirements.txt
```

3.4 Navigate to the App Directory

Use the `cd` command to go to the folder where `app.py` is located. For example:
```python
cd /path/to/your/project/folder
```
3.5 Start the app by running:
```python
streamlit run app.py
```
After launching, Streamlit will provide a local URL, open this link in your web browser to interact with the application.



