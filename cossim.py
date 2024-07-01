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
    # Check for zero vectors to avoid division by zero errors
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Define file paths
movie_annotations_path = 'C:/Users/mohamedm/Networkintegration/sherlock/data/fmri/templates/Sherlock_annotations_by_events.csv'
recall_folder_path = 'C:/Users/mohamedm/Networkintegration/sherlock/data/fmri/templates/recall_transcripts_clean'
output_path = 'C:/Users/mohamedm/Networkintegration/sherlock/data/fmri/templates/annotation_similarities_zeros.csv'

# Load the movie annotations
movie_annotations_df = pd.read_csv(movie_annotations_path)

# Check the columns of movie_annotations_df
print("Movie annotations columns:", movie_annotations_df.columns)

# Initialize a list to store similarities for each subject
all_similarities = []

# Process each recall transcript CSV file
for subject_number in range(1, 18):
    recall_file = f'sub-{subject_number:02d}_recall_transcript.csv'
    recall_path = os.path.join(recall_folder_path, recall_file)

    if os.path.exists(recall_path):
        recall_df = pd.read_csv(recall_path)

        # Check the columns of recall_df
        print(f"Recall annotations columns for subject {subject_number}:", recall_df.columns)

        # Merge movie annotations and recall annotations on event number
        merged_df = pd.merge(movie_annotations_df, recall_df, left_on='event', right_on='events', how='inner')

        # Check the columns of merged_df
        print(f"Merged dataframe columns for subject {subject_number}:", merged_df.columns)

        # Iterate through the merged dataframe and compute cosine similarities
        for _, row in merged_df.iterrows():
            movie_annotation = row['annotations']  # Use the correct column name for movie annotations
            recall_annotation = row['transcript']  # Use the correct column name for recall annotations

            # Skip computation if recall_annotation is NaN
            if pd.isnull(recall_annotation):
                similarity = 0.0
            else:
                # Get embeddings
                movie_embedding = get_embeddings([movie_annotation])[0]
                recall_embedding = get_embeddings([recall_annotation])[0]

                # Compute similarity
                similarity = cosine_similarity(movie_embedding, recall_embedding)

            # Append result to the list
            all_similarities.append({
                'subject': f'sub-{subject_number:02d}',
                'event': row['event'],
                'similarity': similarity
            })

# Convert the list to a DataFrame
similarities_df = pd.DataFrame(all_similarities)

# Save the DataFrame to a CSV file
similarities_df.to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
