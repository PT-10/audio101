import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv("./data/output/final_output_with_prediction.csv")

# Assuming the file has columns 'true_label' and 'predicted_score'
true_labels = data['label']
predicted_scores = data['score']

# Function to calculate FAR and FRR
def calculate_far_frr(threshold, true_labels, predicted_scores):
    # FAR: False Acceptance Rate
    far = np.sum((predicted_scores >= threshold) & (true_labels == 0)) / np.sum(true_labels == 0)
    # FRR: False Rejection Rate
    frr = np.sum((predicted_scores < threshold) & (true_labels == 1)) / np.sum(true_labels == 1)
    return far, frr

# Calculate EER
thresholds = np.sort(predicted_scores)
fars = []
frrs = []
for threshold in thresholds:
    far, frr = calculate_far_frr(threshold, true_labels, predicted_scores)
    fars.append(far)
    frrs.append(frr)

# Find the threshold where FAR and FRR are closest
fars = np.array(fars)
frrs = np.array(frrs)
eer_index = np.nanargmin(np.abs(fars - frrs))
eer = (fars[eer_index] + frrs[eer_index]) / 2
eer_threshold = thresholds[eer_index]

print(f"Equal Error Rate (EER): {eer}")
print(f"EER Threshold: {eer_threshold}")

# Calculate TAR@1%FAR
# Find the threshold for 1% FAR
far_1_percent_threshold = thresholds[np.where(fars <= 0.01)[0][0]]

# Calculate TAR at this threshold
tar_at_1_percent_far = np.sum((predicted_scores >= far_1_percent_threshold) & (true_labels == 1)) / np.sum(true_labels == 1)

print(f"TAR@1%FAR: {tar_at_1_percent_far}")