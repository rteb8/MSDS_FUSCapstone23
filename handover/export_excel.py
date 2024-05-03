import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# path where the model and tokenizer were saved
model_save_path = './fusBERT.pt'
tokenizer_save_path = './tokenizer'

###################################

# initialize the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)

# load the model's state_dict
model.load_state_dict(torch.load(model_save_path))

model.eval()

###################################

def predict_abstract(abstract):
    # Ensure the model is on the correct device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Tokenize the input abstract
    inputs = tokenizer(abstract, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # Move the tokenized inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Print the raw logits
    # print("Raw logits:", logits)
    
    # Convert logits to probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)
    
    # Move predictions back to CPU for easy handling (if they were on GPU)
    confidence = confidence.cpu().item() * 100  # as percentage
    prediction = prediction.cpu().item()  # binary indication
    
    return prediction, confidence, logits.cpu().numpy()

###################################

# read in excel sheet
import pandas as pd
import openpyxl

file_path = 'Publication_data_2024-04-01.xlsx' #update file path to relevant file
df = pd.read_excel(file_path)

# Prediction and confidence assignment
predictions, confidences, raw_logits = [], [], []

df['Abstract'] = df['Abstract'].dropna()
df['Abstract'] =df['Abstract'].astype(str)
df['Abstract'] = [x.lower() for x in df['Abstract']]
df = df.drop_duplicates(subset='Abstract', keep='first')

for abstract in df['Abstract'].astype(str):
    prediction, confidence, logits = predict_abstract(abstract)
    predictions.append(prediction)
    confidences.append(confidence)
    raw_logits.append(logits.tolist())  # Convert numpy array to list for DataFrame compatibility

# Add the new data to the DataFrame
df['Prediction'] = predictions
df['Confidence'] = confidences
df['Logits'] = raw_logits


# Update DataFrame
df['Prediction'] = predictions
df['Confidence'] = confidences

# Save to a new Excel file
output_file_path = 'updated_Publication_data_2024-04-01.xlsx'
df.to_excel(output_file_path, index=False)

###################################
