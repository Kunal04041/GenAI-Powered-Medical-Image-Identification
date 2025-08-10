import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from tqdm.auto import tqdm
import os
import json
import time
import warnings
import io

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Medical Image Classification System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageExtractor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    def extract_from_pdf(self, pdf_file, output_dir="extracted_images", dpi=300):
        """Extract images from PDF using pdf2image"""
        try:
            from pdf2image import convert_from_bytes
        except ImportError:
            st.error(" pdf2image not installed. Run: apt-get install poppler-utils")
            return []

        os.makedirs(output_dir, exist_ok=True)
        extracted_images = []

        try:
            # Convert PDF pages to images
            pages = convert_from_bytes(pdf_file.read(), dpi=dpi)

            for i, page in enumerate(pages):
                image_path = os.path.join(output_dir, f"pdf_page_{i+1}.jpg")
                page.save(image_path, 'JPEG')
                extracted_images.append(image_path)

            st.success(f" Extracted {len(extracted_images)} images from PDF")
            return extracted_images

        except Exception as e:
            st.error(f" Error extracting from PDF: {str(e)}")
            return []

    def extract_from_url(self, url, output_dir="extracted_images", max_images=50):
        """Extract images from web page using BeautifulSoup"""
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse

        os.makedirs(output_dir, exist_ok=True)
        extracted_images = []

        try:
            # Fetch webpage content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all image tags
            img_tags = soup.find_all('img')
            st.info(f" Found {len(img_tags)} image tags on webpage")

            for i, img in enumerate(img_tags):
                if len(extracted_images) >= max_images:
                    break

                img_url = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if not img_url:
                    continue

                # Handle relative URLs
                img_url = urljoin(url, img_url)

                # Check if it's a valid image format
                if any(fmt in img_url.lower() for fmt in self.supported_formats):
                    try:
                        img_response = requests.get(img_url, headers=headers, timeout=5)
                        if img_response.status_code == 200 and len(img_response.content) > 1000:
                            filename = f"web_image_{len(extracted_images)+1}.jpg"
                            image_path = os.path.join(output_dir, filename)

                            # Convert to PIL Image and save as JPEG
                            img_pil = Image.open(io.BytesIO(img_response.content))
                            if img_pil.mode in ('RGBA', 'LA', 'P'):
                                img_pil = img_pil.convert('RGB')

                            # Resize if too large
                            if img_pil.size[0] > 2048 or img_pil.size[1] > 2048:
                                img_pil.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

                            img_pil.save(image_path, 'JPEG', quality=85)
                            extracted_images.append(image_path)

                    except Exception as e:
                        continue

            st.success(f" Extracted {len(extracted_images)} images from URL")
            return extracted_images

        except Exception as e:
            st.error(f" Error extracting from URL: {str(e)}")
            return []



class MedicalImageClassifier:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = device

    @st.cache_resource
    def load_model(_self):
        """Load BiomedCLIP model"""
        try:
            from open_clip import create_model_from_pretrained, get_tokenizer
        except ImportError:
            st.error(" open_clip_torch not installed properly")
            return False

        with st.spinner(" Loading BiomedCLIP model..."):
            try:
                # Load model and preprocessing
                _self.model, _self.preprocess = create_model_from_pretrained(
                    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                )
                _self.tokenizer = get_tokenizer(
                    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                )

                _self.model.to(_self.device)
                _self.model.eval()

                

                # classification prompts
                _self.templates = [
                    'this is a photo of',
                    'this image shows', 
                    'this is an image of',
                    'a photograph of'
                ]

                _self.medical_labels = [
                    'medical image',
                    'X-ray image',
                    'MRI scan',
                    'CT scan', 
                    'ultrasound image',
                    'histopathology image',
                    'medical photograph',
                    'radiological image',
                    'clinical image',
                    'diagnostic image',
                    'medical chart',
                    'anatomical image'
                ]

                _self.non_medical_labels = [
                    'non-medical image',
                    'natural photograph',
                    'landscape photo',
                    'portrait photo',
                    'everyday object',
                    'architectural photo',
                    'animal photo',
                    'food photo',
                    'abstract image',
                    'general photograph',
                    'street scene',
                    'artwork'
                ]

                return True

            except Exception as e:
                st.error(f" Error loading model: {str(e)}")
                return False

    def classify_single_image(self, image_input):
        """Classify a single image"""
        try:
            # Handle both file paths and PIL images
            if isinstance(image_input, str):
                image = Image.open(image_input)
            else:
                image = image_input

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize if too large
            if image.size[0] > 512 or image.size[1] > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)

            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Prepare text prompts
            all_labels = self.medical_labels + self.non_medical_labels
            text_prompts = []

            for template in self.templates:
                for label in all_labels:
                    text_prompts.append(f"{template} {label}")

            # Tokenize text
            text_tokens = self.tokenizer(text_prompts, context_length=256).to(self.device)

            # Get model predictions
            with torch.no_grad():
                image_features, text_features, logit_scale = self.model(image_tensor, text_tokens)

                # Calculate similarities
                logits = (logit_scale * image_features @ text_features.t()).detach()
                probs = torch.softmax(logits, dim=-1).cpu()

                # Aggregate probabilities by template
                num_templates = len(self.templates)
                num_labels = len(all_labels)

                # Reshape and average across templates
                probs_reshaped = probs.view(num_templates, num_labels)
                avg_probs = torch.mean(probs_reshaped, dim=0)

                # Calculate medical vs non-medical scores
                medical_score = torch.sum(avg_probs[:len(self.medical_labels)]).item()
                non_medical_score = torch.sum(avg_probs[len(self.medical_labels):]).item()

                # Normalize scores
                total_score = medical_score + non_medical_score
                medical_confidence = medical_score / total_score if total_score > 0 else 0.5

                prediction = "medical" if medical_confidence > 0.5 else "non-medical"
                confidence = max(medical_confidence, 1 - medical_confidence)

                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'medical_score': medical_confidence,
                    'non_medical_score': 1 - medical_confidence
                }

        except Exception as e:
            st.error(f" Error classifying image: {str(e)}")
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'medical_score': 0.0,
                'non_medical_score': 0.0
            }

    def classify_batch(self, image_inputs, batch_size=8):
        """Classify multiple images"""
        results = []

        progress_bar = st.progress(0)

        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i:i+batch_size]

            for j, img_input in enumerate(batch):
                result = self.classify_single_image(img_input)
                if isinstance(img_input, str):
                    result['image_path'] = img_input
                    result['image_name'] = os.path.basename(img_input)
                results.append(result)
                progress_bar.progress((i+j+1) / len(image_inputs))

        return results



def evaluate_model(test_data, classifier):
    """Evaluate model on test dataset"""

    # Handle different input formats
    if isinstance(test_data[0], tuple):
        # Format: [(image, label), ...]
        images = [item[0] for item in test_data]
        true_labels = [item[1] for item in test_data]
    else:
        # Format: [{'image': image, 'label': label}, ...]
        images = [item['image'] for item in test_data]
        true_labels = [item['label'] for item in test_data]

    results = classifier.classify_batch(images)
    valid_results = [(r, true_labels[i]) for i, r in enumerate(results) if r['prediction'] != 'error']

    if not valid_results:
        st.error(" No valid predictions to evaluate")
        return None

    predictions = [r[0]['prediction'] for r in valid_results]
    valid_true_labels = [r[1] for r in valid_results]

    precision = precision_score(valid_true_labels, predictions, pos_label='medical')
    recall = recall_score(valid_true_labels, predictions, pos_label='medical')
    cm = confusion_matrix(valid_true_labels, predictions)

    return precision, recall, cm


# Initialize cached components
@st.cache_resource
def get_classifier():
    classifier = MedicalImageClassifier()
    if classifier.load_model():
        return classifier
    else:
        return None

@st.cache_resource
def get_extractor():
    return ImageExtractor()


# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header"> Medical Image Classification System</h1>', 
                unsafe_allow_html=True)

    # Initialize components
    with st.spinner("Initializing system components..."):
        classifier = get_classifier()
        extractor = get_extractor()

    if classifier is None:
        st.error(" Failed to load model. Please check your installation.")
        st.stop()

    
    # Navigation
    st.sidebar.title(" Navigation")
    mode = st.sidebar.selectbox(
        "Choose Task",
        [
            "Single Image Classification",
            "Batch Image Classification", 
            "PDF Image Extraction & Classification",
            "URL Image Extraction & Classification",
            "Model Evaluation"
        ]
    )

    # Single Image Classification
    if mode == "Single Image Classification":
        st.markdown('<h2 class="section-header"> Single Image Classification</h2>', 
                    unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload a medical image", 
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload an image to classify as medical or non-medical"
        )

        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(" Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                st.info(f"Image size: {image.size[0]} x {image.size[1]} pixels")

            with col2:
                st.subheader(" Classification Results")

                with st.spinner("Classifying image..."):
                    result = classifier.classify_single_image(image)

                # Display results
                prediction = result['prediction']
                confidence = result['confidence']

                if prediction == "medical":
                    st.success(f"** Prediction:** Medical Image")
                    st.success(f"**Confidence:** {confidence:.4f} ({confidence*100:.2f}%)")
                elif prediction == "non-medical":
                    st.info(f"** Prediction:** Non-Medical Image")
                    st.info(f"** Confidence:** {confidence:.4f} ({confidence*100:.2f}%)")
                else:
                    st.error(" Classification error occurred")

                # Show detailed scores
                if result['prediction'] != 'error':
                    st.subheader(" Detailed Scores")

                    col_med, col_non = st.columns(2)

                    with col_med:
                        st.metric(" Medical Score", f"{result['medical_score']:.4f}")

                    with col_non:
                        st.metric(" Non-Medical Score", f"{result['non_medical_score']:.4f}")

    # Batch Image Classification
    elif mode == "Batch Image Classification":
        st.markdown('<h2 class="section-header"> Batch Image Classification</h2>', 
                    unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload multiple images", 
            type=['png', 'jpg', 'jpeg', 'bmp'],
            accept_multiple_files=True,
            help="Upload multiple images for batch classification"
        )

        if uploaded_files:
            st.success(f" **{len(uploaded_files)} images uploaded**")

            # Show preview of first few images
            if len(uploaded_files) > 0:
                st.subheader(" Image Preview")
                preview_cols = st.columns(min(4, len(uploaded_files)))
                for i, col in enumerate(preview_cols):
                    if i < len(uploaded_files):
                        with col:
                            img = Image.open(uploaded_files[i])
                            st.image(img, caption=uploaded_files[i].name, use_column_width=True)

            if st.button(" Classify All Images", type="primary"):
                images = [Image.open(f) for f in uploaded_files]

                with st.spinner("Processing batch classification..."):
                    results = classifier.classify_batch(images)

                # Create results dataframe
                df_data = []
                for i, (result, file) in enumerate(zip(results, uploaded_files)):
                    df_data.append({
                        'filename': file.name,
                        'prediction': result['prediction'],
                        'confidence': f"{result['confidence']:.4f}",
                        'medical_score': f"{result['medical_score']:.4f}",
                        'non_medical_score': f"{result['non_medical_score']:.4f}"
                    })

                df = pd.DataFrame(df_data)

                # Display results
                st.subheader(" Batch Classification Results")
                st.dataframe(df, use_container_width=True)

                # Summary statistics
                medical_count = (df['prediction'] == 'medical').sum()
                non_medical_count = (df['prediction'] == 'non-medical').sum()
                error_count = (df['prediction'] == 'error').sum()

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(" Medical Images", medical_count)

                with col2:
                    st.metric(" Non-Medical Images", non_medical_count)

                with col3:
                    st.metric(" Errors", error_count)

                with col4:
                    avg_confidence = df['confidence'].astype(float).mean()
                    st.metric(" Avg Confidence", f"{avg_confidence:.4f}")

                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label=" Download Results CSV",
                    data=csv,
                    file_name="classification_results.csv",
                    mime="text/csv"
                )

    # PDF Image Extraction & Classification
    elif mode == "PDF Image Extraction & Classification":
        st.markdown('<h2 class="section-header"> PDF Image Extraction & Classification</h2>', 
                    unsafe_allow_html=True)

        uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

        if uploaded_pdf is not None:
            st.info(f" PDF uploaded: {uploaded_pdf.name} ({uploaded_pdf.size} bytes)")

            if st.button(" Extract and Classify Images", type="primary"):
                with st.spinner("Extracting images from PDF..."):
                    image_paths = extractor.extract_from_pdf(uploaded_pdf)

                if image_paths:
                    # Classify extracted images
                    with st.spinner("Classifying extracted images..."):
                        results = classifier.classify_batch(image_paths)

                    # Display results
                    df_data = []
                    for i, (result, path) in enumerate(zip(results, image_paths)):
                        df_data.append({
                            'page': f"Page {i+1}",
                            'prediction': result['prediction'],
                            'confidence': f"{result['confidence']:.4f}",
                            'medical_score': f"{result['medical_score']:.4f}",
                            'non_medical_score': f"{result['non_medical_score']:.4f}"
                        })

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)

                    # Summary
                    medical_count = (df['prediction'] == 'medical').sum()
                    non_medical_count = (df['prediction'] == 'non-medical').sum()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(" Medical Images", medical_count)
                    with col2:
                        st.metric(" Non-Medical Images", non_medical_count)
                else:
                    st.warning(" No images could be extracted from the PDF")

    # URL Image Extraction & Classification  
    elif mode == "URL Image Extraction & Classification":
        st.markdown('<h2 class="section-header"> URL Image Extraction & Classification</h2>', 
                    unsafe_allow_html=True)

        url = st.text_input("Enter webpage URL", placeholder="https://example.com/medical-images")
        max_images = st.slider("Maximum images to extract", 1, 100, 20)

        if url and st.button(" Extract and Classify Images", type="primary"):
            if url.startswith(('http://', 'https://')):
                with st.spinner("Extracting images from URL..."):
                    image_paths = extractor.extract_from_url(url, max_images=max_images)

                if image_paths:
                    # Classify extracted images
                    with st.spinner("Classifying extracted images..."):
                        results = classifier.classify_batch(image_paths)

                    # Display results
                    df_data = []
                    for i, (result, path) in enumerate(zip(results, image_paths)):
                        df_data.append({
                            'image': f"Image {i+1}",
                            'prediction': result['prediction'],
                            'confidence': f"{result['confidence']:.4f}",
                            'medical_score': f"{result['medical_score']:.4f}",
                            'non_medical_score': f"{result['non_medical_score']:.4f}"
                        })

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)

                    # Summary
                    medical_count = (df['prediction'] == 'medical').sum()
                    non_medical_count = (df['prediction'] == 'non-medical').sum()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(" Medical Images", medical_count)
                    with col2:
                        st.metric("Non-Medical Images", non_medical_count)
                else:
                    st.warning(" No images could be extracted from the URL")
            else:
                st.error(" Please enter a valid URL starting with http:// or https://")

    # Model Evaluation
    elif mode == "Model Evaluation":
        st.markdown('<h2 class="section-header"> Model Evaluation</h2>', 
                    unsafe_allow_html=True)

        st.info(" Upload medical and non-medical images to evaluate model performance with precision, recall, and confusion matrix.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Medical Images")
            medical_files = st.file_uploader(
                "Upload Medical Images", 
                type=['png', 'jpg', 'jpeg', 'bmp'],
                accept_multiple_files=True,
                key="medical"
            )
            if medical_files:
                st.success(f" {len(medical_files)} medical images uploaded")

        with col2:
            st.subheader(" Non-Medical Images") 
            non_medical_files = st.file_uploader(
                "Upload Non-Medical Images", 
                type=['png', 'jpg', 'jpeg', 'bmp'],
                accept_multiple_files=True,
                key="non_medical"
            )
            if non_medical_files:
                st.success(f" {len(non_medical_files)} non-medical images uploaded")

        if medical_files and non_medical_files:
            if st.button(" Evaluate Model", type="primary"):
                # Prepare test data
                test_data = []

                # Add medical images with labels
                for file in medical_files:
                    test_data.append((Image.open(file), 'medical'))

                # Add non-medical images with labels  
                for file in non_medical_files:
                    test_data.append((Image.open(file), 'non-medical'))

                # Evaluate model
                with st.spinner(" Evaluating model performance..."):
                    result = evaluate_model(test_data, classifier)

                if result is not None:
                    precision, recall, cm = result

                    # Display metrics
                    st.subheader(" Evaluation Results")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(" Precision", f"{precision:.4f}")
                    with col2:
                        st.metric(" Recall", f"{recall:.4f}")

                    # Display confusion matrix
                    st.subheader(" Confusion Matrix")

                    # Create matplotlib figure
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                               xticklabels=['Non-Medical', 'Medical'],
                               yticklabels=['Non-Medical', 'Medical'],
                               ax=ax)
                    ax.set_title('Confusion Matrix')
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')

                    st.pyplot(fig)

                    # Print results (matching your original Colab format)
                    st.success(" Model evaluation completed!")
                    st.code(f"""
Precision: {precision:.4f}
Recall: {recall:.4f}
Confusion Matrix:
{cm}
                    """)

    # Footer
    st.markdown("---")
    
    

if __name__ == "__main__":
    main()
