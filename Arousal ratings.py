import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("stabilityai/StableBeluga-13B")
model = AutoModelForSequenceClassification.from_pretrained("stabilityai/StableBeluga-13B")

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Sample texts for prediction
texts = ["I'm feeling very excited!", "I'm so bored right now.", "This is an amazing experience!"]

# Preprocess the texts
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU if available

# Run predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Assuming the second label is the arousal rating
arousal_ratings = predictions[:, 1].tolist()

# Print the arousal ratings
print("Arousal Ratings:", arousal_ratings)
