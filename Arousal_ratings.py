import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("stabilityai/StableBeluga-13B")
model = AutoModelForSequenceClassification.from_pretrained("stabilityai/StableBeluga-13B")

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Directory containing the CSV files
csv_dir = 'C:/Users/mohamedm/Networkintegration/filmfestival/data/fmri/templates/1_transcripts'

# Directory to save the results
output_dir = 'C:/Users/mohamedm/Networkintegration/filmfestival/data/fmri/templates/results'
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file in the directory
for csv_file in os.listdir(csv_dir):
    if csv_file.endswith('.csv'):
        csv_file_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_file_path)
        texts = df['transcript'].tolist()

        # Preprocess the texts
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU if available

        # Run predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Assuming the second label is the arousal rating
        arousal_ratings = predictions[:, 1].tolist()

        # Save the arousal ratings to a new DataFrame
        df['arousal_rating'] = arousal_ratings
        output_file_path = os.path.join(output_dir, csv_file)
        df.to_csv(output_file_path, index=False)

        print(f"Results saved to {output_file_path}")
