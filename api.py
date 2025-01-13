from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import torch
from fastapi.middleware.cors import CORSMiddleware
import boto3
from pdf2image import convert_from_bytes
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
)

# AWS Textract client
client = boto3.client('textract', region_name='us-east-1')

# Load the tokenizer and model from local path
tokenizer = AutoTokenizer.from_pretrained(
    "C:/Users/MCC/OneDrive - Balqa Applied University/Desktop/fastapi_project/clinicalbert_finetuned"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "C:/Users/MCC/OneDrive - Balqa Applied University/Desktop/fastapi_project/clinicalbert_finetuned"
)
model.eval()

# Define class names for predictions
class_names = ["BLCA", "BRCA", "CESC", "COAD", "GBM", "HNSC", "KIRC", "KIRP",
               "LGG", "LIHC", "LUAD", "LUSC", "OV", "PRAD", "SARC", "STAD", "THCA", "UCEC"]

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=class_names)

# Define device (CPU or GPU)
device = "cpu"
model.to(device)

# Function to extract text from PDF using AWS Textract with concurrency
def extract_text_from_pdf(pdf_bytes):
    try:
        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes)

        # Use a ThreadPoolExecutor for parallel text extraction
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_image, images))

        return "\n".join(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Textract processing: {e}")

# Function to detect QC tables or handwritten content
def is_qc_table_or_handwritten(block):
    # List of keywords and phrases to detect QC table or handwritten content
    qc_keywords = [
        "Diagnosi : Discrepancy",
        "Primary Tumor Site Discrepancy",
        "HIPAA Discrepancy",
        "Priot Malignancy history",
        "Dual/Syrchronous i'rimary Noted",
        "Case is (Aircte)",
        "Yes", "No",  
        "QUANFIED",
        "fyate,iewuwed:",   
        "BISQUALFIED QC TABLE",
        "ICD-0-3",
        
    ]
    
    # Loop through each keyword to check if it exists in the block text (case insensitive)
    for keyword in qc_keywords:
        if keyword.lower() in block['Text'].lower():
            return True
    
    

    if block.get('BlockType') == 'WORD' and block.get('Text') == 'handwritten':  # Example of a check for "handwritten"
        return True
    
    return False

# Process individual image using AWS Textract
def process_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # Call AWS Textract for each image
    response = client.detect_document_text(Document={'Bytes': image_bytes})
    extracted_text = ""
    blocks = response.get('Blocks', [])

    # Remove table and handwritten content by filtering out unwanted blocks
    for block in blocks:
        if block['BlockType'] == 'LINE':
            # For simplicity, removing lines that might be from a QC table or handwritten
            if is_qc_table_or_handwritten(block):
                continue
            extracted_text += block['Text'] + "\n"
    
    return extracted_text

# Prediction function for LIME (single-instance processing)
def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return probs.cpu().numpy()

# FastAPI endpoint to process PDF and return predictions with LIME explanation
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read PDF and extract text using AWS Textract
    pdf_bytes = await file.read()
    text = extract_text_from_pdf(pdf_bytes)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")

    # Truncate text to 512 characters for the model
    truncated_text = text[:512]

    # Perform prediction
    inputs = tokenizer([truncated_text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()

    predicted_class = class_names[prediction]

    # Generate LIME explanation targeting the predicted class with optimizations
    try:
        exp = explainer.explain_instance(
            truncated_text,
            predict_proba,
            num_features=10,       
            labels=[prediction],  
            num_samples=150      
        )
        explanation_html = exp.as_html()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LIME explanation: {e}")

    return {"predicted_class": predicted_class, "explanation": explanation_html}

# Main entry point to run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)