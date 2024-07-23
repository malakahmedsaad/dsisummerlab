from pathlib import Path
import pandas as pd
import numpy as np
import os
import csv
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import submitit  # Importing submitit for SLURM job submission

def process_annotations(path_to_csv: str, output_path: str, model_path: str, subject_id: str):
    """ Process arousal annotations and save ratings to CSV """

    # Define paths and filenames
    annot_csv = f"sub-{subject_id}_recall_concat.csv"  ## input file
    ratings_csv = f"sub-{subject_id}_arousal_ratings.csv"  ## output file
    filename = os.path.join(path_to_csv, annot_csv)
    savefile = os.path.join(output_path, ratings_csv)

    # Get annotation csv
    df = pd.read_csv(filename)
    df = df.dropna(subset=['transcript'])  ## drop rows with NAN in the column subset

    # Define column names
    column_name = ['event_number', 'rater_1']

    # Define input prompt
    msg = "Arousal refers to when you are feeling very mentally or physically alert, \
        activated, and/or energized. Read the following description of a scene \
        and rate the arousal level of the scene on a scale of 1 to 10, \
        with 1 being low arousal and 10 being high arousal. Please only give a numeric \
        rating."

    # Initialize tokenizer and model
    print("[+] Initializing Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print("[+] Finished Initializing Tokenizer")

    print("[+] Initializing Model")
    my_model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    print("[+] Finished Initializing Model")

    # Write arousal ratings to file dynamically
    with open(savefile, 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n',)
        dict_ratings = csv.DictWriter(csv_file, fieldnames=column_name)
        dict_ratings.writeheader()

        # For each event:
        for index, row in df.iterrows():
            dict_row = {"event_number": row['event_number']}

            # Fetch annotation for this event
            thisEvent = row['events']
            print(f"[+] Processing event number: {thisEvent}")

            # Fetch annotation for this event
            thisScene = row['transcript']

            system_prompt = "### System:\nYou are Stable Beluga 13B, an AI that follows instructions extremely well. \
            Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"
            message = f"{msg} \nScene: {thisScene}"
            thisMessageLength = round(len(message))

            prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"
            inputs = tokenizer(prompt, return_tensors="pt")
            output = my_model.generate(**inputs, do_sample=True, top_p=0.85, top_k=0, max_length=thisMessageLength)

            # Print inputs and outputs
            print(tokenizer.decode(output[0], skip_special_tokens=True))
            result = tokenizer.decode(output[0], skip_special_tokens=True)

            # Grab arousal rating from the output
            arousal = re.findall(r'### Assistant:\s*(.*?)\s*(?=###|$)', result, re.DOTALL)

            # Make sure there is only one rating in the output
            assert len(arousal) == 1

            # Convert output to int
            a_rating = ''.join(x for x in arousal if x.isdigit())
            dict_row["rater_1"] = a_rating

            # Append rating to csv
            dict_ratings.writerow(dict_row)

def main(query_file):
    """ Main function to process annotations based on query parameters """

    # Read in query from JSON file
    with open(query_file) as f:
        query = json.load(f)

    # Get parameters from query
    path_to_csv = query.get("path_to_csv")
    output_directory = Path(query.get("output_directory")).resolve()
    model_path = query.get("model_path")

    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Initialize submitit executor
    executor = submitit.AutoExecutor(folder=output_directory)
    executor.update_parameters(**query.get("slurm", {}))

    # Submit job using submitit for each subject
    for subject_id in range(1):
        subject_id_str = f"{subject_id:02}"  # Format as two-digit string
        if query.get("submitit", False):
            executor.submit(process_annotations, path_to_csv, str(output_directory), model_path, subject_id_str)
        else:
            process_annotations(path_to_csv, str(output_directory), model_path, subject_id_str)

if __name__ == "__main__":
    import argparse
    import json

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process arousal annotations.")
    parser.add_argument("--query", help="Path to JSON query file", required=True)
    args = parser.parse_args()

    # Ensure query file exists
    query_path = Path(args.query).resolve()
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {args.query}")

    # Call main function with query file
    main(query_path)
    # "slurm": {
    #     "slurm_partition": "general",
    #     "slurm_job_name": "sample",
    #     "slurm_nodes": 1,
    #     "slurm_time": "5:00",
    #     "slurm_gres": "gpu:1"
    # }