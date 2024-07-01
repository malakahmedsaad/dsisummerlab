import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

# Load the Universal Sentence Encoder
model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(model_url)


def get_embeddings(texts):
    embeddings = embed(texts)
    return embeddings.numpy()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



movie_annotations_path = 'C:/Users/mohamedm/Networkintegration/filmfestival/data/fmri/templates/filmfest_annotations_KG.csv'
movie_annotations_df = pd.read_csv(movie_annotations_path)
movie_annotations = movie_annotations_df['annotation'].dropna().tolist()

# Get movie annotations embeddings
movie_annotation_embeddings = get_embeddings(movie_annotations)

# Folder containing recall transcripts CSV files
recall_folder_path = '../../Networkintegration/filmfestival/data/fmri/templates/1_transcripts'

# Initialize a dictionary to store similarities for each subject
all_similarities = {}

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
        for movie_embedding in movie_annotation_embeddings:
            event_similarities = []
            for recall_embedding in recall_annotation_embeddings:
                similarity = cosine_similarity(movie_embedding, recall_embedding)
                event_similarities.append(similarity)
            subject_similarities.append(event_similarities)


        subject_number = recall_file.split('_')[0]

        # Store the similarities for this subject
        all_similarities[subject_number] = subject_similarities

# Output the similarities for inspection
for subject, similarities in all_similarities.items():
    print(f"Subject: {subject}")
    for i, event_similarities in enumerate(similarities):
        print(f"  Movie Annotation {i + 1} similarities: {event_similarities}")
