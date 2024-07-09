import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("stabilityai/StableBeluga-13B", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("stabilityai/StableBeluga-13B", torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True, device_map="auto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_dir = 'C:/Users/mohamedm/Networkintegration/filmfestival/data/fmri/templates/1_transcripts'

output_dir = 'C:/Users/mohamedm/Networkintegration/filmfestival/data/fmri/templates/results'
os.makedirs(output_dir, exist_ok=True)

system_prompt = "### System:\nThis is a system prompt, please behave and help the user.\n\n"

for csv_file in os.listdir(input_dir):
    if csv_file.endswith('.csv'):
        csv_file_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(csv_file_path)
        texts = df['transcript'].tolist()

        # Preprocess the texts and generate prompts
        arousal_ratings = []
        for text in texts:
            user_message = f"Analyze the sentiment of the following text:\n\n{text}"
            prompt = f"{system_prompt}### User:\n{user_message}\n\n### Assistant:\n"

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)

            result = tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract arousal rating from the result
            arousal_rating = float(result.split()[-1])  # Replace this with actual logic to extract rating
            arousal_ratings.append(arousal_rating)

        df['arousal_rating'] = arousal_ratings
        output_file_path = os.path.join(output_dir, csv_file)
        df.to_csv(output_file_path, index=False)

        print(f"Results saved to {output_file_path}")
