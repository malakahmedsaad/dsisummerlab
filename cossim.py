import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

# Load the Universal Sentence Encoder
model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(model_url)

def get_embeddings(texts):
    # Convert texts to embeddings
    embeddings = embed(texts)
    return embeddings.numpy()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Define file paths
movie_annotations_path = 'C:/Users/mohamedm/Networkintegration/filmfestival/data/fmri/templates/filmfest_annotations_KG.csv'
recall_folder_path = 'C:/Users/mohamedm/Networkintegration/filmfestival/data/fmri/templates/1_transcripts'
output_path = 'C:/Users/mohamedm/Networkintegration/filmfestival/data/fmri/templates/annotation_similarities.csv'

# Load the movie annotations
movie_annotations_df = pd.read_csv(movie_annotations_path)
movie_annotations = movie_annotations_df['annotation'].dropna().tolist()

# Get movie annotations embeddings
movie_annotation_embeddings = get_embeddings(movie_annotations)

# Initialize a list to store similarities for each subject
all_similarities = []

# Process each recall transcript CSV file
for recall_file in os.listdir(recall_folder_path):
    if recall_file.endswith('.csv'):
        recall_path = os.path.join(recall_folder_path, recall_file)
        recall_df = pd.read_csv(recall_path)

        # Extract non-null transcripts from the recall annotations
        recall_annotations = recall_df['transcript'].dropna().tolist()

        # Get recall annotations embeddings
        recall_annotation_embeddings = get_embeddings(recall_annotations)

        # Compute cosine similarity between each movie and recall annotation
        subject_similarities = []
        for movie_index, movie_embedding in enumerate(movie_annotation_embeddings):
            for recall_index, recall_embedding in enumerate(recall_annotation_embeddings):
                similarity = cosine_similarity(movie_embedding, recall_embedding)
                # Append result to the list
                all_similarities.append({
                    'subject': recall_file.split('_')[0],
                    'movie_annotation_index': movie_index + 1,
                    'recall_annotation_index': recall_index + 1,
                    'similarity': similarity
                })

# Convert the list to a DataFrame
similarities_df = pd.DataFrame(all_similarities)

# Save the DataFrame to a CSV file
similarities_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
