# Leveraging-ClinicalBERT-for-Histopathology-Cancer-Reports-Classification-
Graduation Project

# Leveraging ClinicalBERT for Histopathology Cancer Reports Classification

## Project Overview
This project leverages **ClinicalBERT**, a pre-trained transformer-based NLP model, to classify unstructured pathology reports from **The Cancer Genome Atlas (TCGA)** into 18 distinct cancer types. By integrating Explainable AI (XAI) techniques like **LIME**, this system provides both accurate predictions and interpretable results, critical for clinical decision-making.

---

## Features
- Automated classification of pathology reports into cancer types.
- Explainable AI (XAI) integration using **LIME** for interpretability.
- High accuracy with metrics:
  - Training Accuracy: 98.5%
  - Validation Accuracy: 97.5%
  - Test Accuracy: 97.1%
- Backend API for real-time predictions using **FastAPI**.
- User-friendly HTML frontend for uploading reports and viewing results.

---

## Requirements

### Software
- Python 3.10 or later
- Libraries:
  - `transformers`
  - `torch`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `lime`
  - `fastapi`
  - `uvicorn`
  - `boto3`
  - `matplotlib`
  - `pdf2image`

### Hardware
- **GPU**: For model fine-tuning (Google Colab GPU recommended).
- **CPU**: For API development and AWS Textract integration.
- **RAM**: Minimum 16 GB for local processing.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo-name/clinicalbert-pathology-classification.git
   cd clinicalbert-pathology-classification
