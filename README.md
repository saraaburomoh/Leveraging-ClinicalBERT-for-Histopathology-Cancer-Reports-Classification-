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

## References

### Articles and Literature
1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?" Explaining the Predictions of Any Classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. [Paper Link](https://dl.acm.org/doi/10.1145/2939672.2939778)
2. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*. [Paper Link](https://arxiv.org/abs/1706.03762)
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*. [Paper Link](https://arxiv.org/abs/1810.04805)
4. Lee, J., Yoon, W., Kim, S., et al. (2019). BioBERT: A Pre-trained Biomedical Language Representation Model for Biomedical Text Mining. *Bioinformatics*. [Paper Link](https://arxiv.org/abs/1901.08746)
5. Huang, K., Altosaar, J., & Ranganath, R. (2020). ClinicalBERT: Improving Predictions of Hospital Readmissions with Clinical Notes. *Journal of the American Medical Informatics Association (JAMIA)*. [Paper Link](https://academic.oup.com/jamia/article/27/9/1464/5891901)

### Data Sources
1. The Cancer Genome Atlas (TCGA). GDC Data Portal. [Website](https://portal.gdc.cancer.gov/)

### XAI Tools
1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). LIME (Local Interpretable Model-agnostic Explanations). *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. [Tool Overview](https://github.com/marcotcr/lime)

