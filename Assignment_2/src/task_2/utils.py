# from utils import *
# audio_path = "D:/Study/4th Year/8th Semester/Speech Understanding/Assignment 2/src/data/Language Detection Dataset/Punjabi/1887.mp3"
# mfcc, sr = extract_mfcc(audio_path, n_mfcc=13)
# print(mfcc)

# import pandas as pd
# import ast

files = ["data/features/Bengali_mfcc_features.csv","data/features/Gujarati_mfcc_features.csv","data/features/Hindi_mfcc_features.csv",
         "data/features/Kannada_mfcc_features.csv","data/features/Malayalam_mfcc_features.csv","data/features/Marathi_mfcc_features.csv","data/features/Punjabi_mfcc_features.csv",
         "data/features/Tamil_mfcc_features.csv","data/features/Telugu_mfcc_features.csv","data/features/Urdu_mfcc_features.csv",]

# # Create DataFrame
# df = pd.read_csv(files[9])

# # Convert string representations of lists to actual lists
# df["mean"] = df["mean"].apply(ast.literal_eval)
# df["variance"] = df["variance"].apply(ast.literal_eval)

# # Expand the mean and variance columns into separate columns
# mean_df = pd.DataFrame(df["mean"].tolist(), columns=[f"mfcc_{i}_mean" for i in range(len(df["mean"][0]))])
# variance_df = pd.DataFrame(df["variance"].tolist(), columns=[f"mfcc_{i}_var" for i in range(len(df["variance"][0]))])

# # Concatenate the original DataFrame with the new columns
# result = pd.concat([df.drop(columns=["mean", "variance"]), mean_df, variance_df], axis=1)

# result.to_csv("data/features/Urdu_mfcc_features.csv", index=False)
# print("Saved expanded features to expanded_features.csv")

import pandas as pd
import glob

import pandas as pd
import random

# List of file paths (replace with your actual file paths)
# file_paths = [
#     "file1.csv",
#     "file2.csv",
#     "file3.csv",
#     "file4.csv",
#     "file5.csv",
#     "file6.csv",
#     "file7.csv",
#     "file8.csv",
#     "file9.csv",
#     "file10.csv"
# ]

# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Loop through each file
for file_path in files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    df['source_file'] = file_path.split('/')[-1]
    # Randomly sample 5000 rows from the DataFrame
    sampled_df = df.sample(n=5000, random_state=42)  # Set random_state for reproducibility
    
    # Append the sampled data to the combined DataFrame
    combined_data = pd.concat([combined_data, sampled_df], ignore_index=True)

# Save the combined data to a new CSV file
combined_data.to_csv("data/features/combined_data.csv", index=False)

print("Combined CSV file created successfully!")