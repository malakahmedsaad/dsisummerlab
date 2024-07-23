import pandas as pd
import numpy as np

# Define subject IDs
subject_ids = ["sub-01", "sub-02", "sub-07", "sub-08", "sub-09", "sub-10",
               "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
               "sub-18", "sub-19", "sub-20"]

# Load recall events from file
filmfest_recall_events = pd.read_mat('filmfest_recall_events.mat')

# Placeholder function to get recall events for a subject
def get_recall_events(subject_id):
    subject_number = int(subject_id.split('-')[1])
    recall_events = filmfest_recall_events['boundaries']['recall'][f'{subject_number}']['recall_events']
    sorted_events = sorted(enumerate(recall_events), key=lambda x: x[1])
    sorted_indices = [i for i, _ in sorted_events]
    return sorted_events, sorted_indices

# Load swapped boundaries data from file
swapped_boundaries_avg = pd.read_mat('swapped_boundaries_avg.mat')

# Placeholder function to get hippocampus boundary data for a subject
def get_hippocampus_boundary_data(subject_id, event_indices):
    subject_number = int(subject_id.split('-')[1])
    movie_boundary = swapped_boundaries_avg['hippocampus']['movie'][f'{subject_number}']
    recall_boundary = swapped_boundaries_avg['hippocampus']['recall'][f'{subject_number}']
    movie_boundary_sorted = [movie_boundary[i] for i in event_indices]
    recall_boundary_sorted = [recall_boundary[i] for i in event_indices]
    return movie_boundary_sorted, recall_boundary_sorted

# Initialize an empty DataFrame
columns = ["subject",
